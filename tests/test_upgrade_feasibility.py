"""End-to-end tests for ``POST /analyze/upgrade-feasibility``.

Each candidate is solved independently with a temporary exact-version root
pin on the target; the original roots and locks still participate, so
feasibility is always a full-closure backtracking solve.  Hitting
``max_attempts`` is reported as ``undetermined``, never as a conflict.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _feasibility(**payload):
    response = client.post("/analyze/upgrade-feasibility", json=payload)
    return response.status_code, response.json()


PACKAGES = {
    "app": {"1.0": {"dependencies": {"web": ">=2.0", "db": ""}}},
    "web": {
        "2.0": {"dependencies": {"common": "<2.0"}},
        "3.0": {"dependencies": {"common": ">=2.0"}},
    },
    "db": {"1.0": {"dependencies": {"common": "==1.5"}}},
    "common": {"1.5": {}, "2.0": {}},
}


# ---------------------------------------------------------------------------
# Happy path and result shape
# ---------------------------------------------------------------------------


def test_candidates_are_reported_in_input_order_with_statuses():
    status, body = _feasibility(
        packages=PACKAGES,
        root={"app": ""},
        locks={"web": "2.0", "common": "1.5"},
        target="web",
        candidates=["2.0", "3.0"],
    )
    assert status == 200
    assert body["target"] == {"package": "web", "candidates": ["2.0", "3.0"]}
    assert body["allow_prereleases"] is False
    assert body["summary"] == {
        "candidates": 2, "ok": 1, "conflict": 1, "undetermined": 0
    }

    baseline = body["baseline"]
    assert baseline["status"] == "ok"
    assert baseline["resolved"] == {
        "app": "1.0", "common": "1.5", "db": "1.0", "web": "2.0"
    }
    assert baseline["locks"] == {"common": "applied", "web": "applied"}

    entries = body["candidates"]
    assert [entry["version"] for entry in entries] == ["2.0", "3.0"]

    ok_entry = entries[0]
    assert ok_entry["status"] == "ok"
    assert ok_entry["attempts"] > 0
    assert ok_entry["resolved"] == baseline["resolved"]
    assert ok_entry["locks"] == {"common": "applied", "web": "applied"}
    assert ok_entry["changes"] == {"added": [], "removed": [], "changed": []}
    assert "conflict" not in ok_entry

    bad_entry = entries[1]
    assert bad_entry["status"] == "conflict"
    assert bad_entry["attempts"] > 0
    assert "resolved" not in bad_entry
    terminals = bad_entry["conflict"]["terminal_conflicts"]
    assert terminals[0]["package"] == "common"
    assert terminals[0]["reason"] == "unsatisfiable"
    # The two mutually exclusive requirements on ``common`` are both shown.
    requirements = {
        c["requirement"]
        for c in terminals[0]["active_constraints"]
    }
    assert requirements == {"common==1.5", "common>=2.0"}

    # attempts is the sum of the baseline and every candidate solve.
    assert body["attempts"] == (
        baseline["attempts"] + sum(entry["attempts"] for entry in entries)
    )


def test_successful_candidate_carries_full_resolution_and_changes():
    packages = {
        "app": {"1.0": {"dependencies": {"web": ""}}},
        "web": {
            "3.0": {"dependencies": {"common": ">=2.0", "extra": ""}},
            "2.0": {"dependencies": {"common": "<2.0"}},
        },
        "common": {"1.5": {}, "2.0": {}},
        "extra": {"1.0": {}},
    }
    status, body = _feasibility(
        packages=packages,
        root={"app": ""},
        target="web",
        candidates=["2.0"],
    )
    assert status == 200
    # Baseline keeps the highest web 3.0 closure.
    assert body["baseline"]["resolved"] == {
        "app": "1.0", "common": "2.0", "extra": "1.0", "web": "3.0"
    }
    entry = body["candidates"][0]
    assert entry["status"] == "ok"
    assert entry["resolved"] == {
        "app": "1.0", "common": "1.5", "web": "2.0"
    }
    assert entry["changes"] == {
        "added": [],
        "removed": [{"package": "extra", "version": "1.0"}],
        "changed": [
            {"package": "common", "from": "2.0", "to": "1.5"},
            {"package": "web", "from": "3.0", "to": "2.0"},
        ],
    }
    edges = {(edge["source"], edge["target"]) for edge in entry["dependency_graph"]}
    assert edges == {
        ("<root>", "app"), ("<root>", "web"),
        ("app", "web"), ("web", "common"),
    }
    assert all(
        edge["target_version"] == entry["resolved"][edge["target"]]
        for edge in entry["dependency_graph"]
    )


def test_feasibility_checks_the_full_closure_not_just_the_target_range():
    # web==3.0 exists and satisfies the target pin, but it contradicts the
    # transitive requirement util==1.0 imposes (via app), and app has no
    # alternate version to backtrack into.
    packages = {
        "app": {"1.0": {"dependencies": {"web": "", "util": "==1.0"}}},
        "web": {"2.0": {}, "3.0": {}},
        "util": {"1.0": {"dependencies": {"web": "<3"}}},
    }
    status, body = _feasibility(
        packages=packages,
        root=["app"],
        target="web",
        candidates=["3.0", "2.0"],
    )
    assert status == 200
    assert [entry["status"] for entry in body["candidates"]] == ["conflict", "ok"]
    terminals = body["candidates"][0]["conflict"]["terminal_conflicts"]
    assert any(
        terminal["package"] == "web" and terminal["reason"] == "unsatisfiable"
        for terminal in terminals
    )


def test_candidate_solves_are_independent():
    # The conflicting candidate must not poison the next one: both are fresh
    # full solves with their own attempt budget.
    packages = {
        "app": {"1.0": {"dependencies": {"web": "<3", "util": "==1.0"}}},
        "web": {"2.0": {}, "3.0": {}},
        "util": {"1.0": {"dependencies": {"web": "<3"}}},
    }
    status, body = _feasibility(
        packages=packages,
        root=["app"],
        target="web",
        candidates=["3.0", "2.0"],
    )
    assert status == 200
    assert [entry["status"] for entry in body["candidates"]] == ["conflict", "ok"]
    # Each candidate carries only its own solve's attempt count.
    assert body["candidates"][0]["attempts"] > 0
    assert body["candidates"][1]["attempts"] > 0


def test_each_candidate_gets_the_full_attempt_budget():
    packages = {
        "app": {"1.0": {"dependencies": {"p0": ""}}},
        "p0": {
            "1.0": {},
            "2.0": {"dependencies": {"p1": ""}},
        },
        "p1": {
            "1.0": {"dependencies": {"missing": ""}},
            "2.0": {"dependencies": {"missing": ""}},
        },
    }
    # Budget exhausts within the infeasible candidate; the other still
    # completes because each candidate runs its own independent solve.
    status, body = _feasibility(
        packages=packages,
        root={"app": ""},
        target="p0",
        candidates=["2.0", "1.0"],
        max_attempts=2,
    )
    assert status == 200
    limited, ok = body["candidates"]
    assert limited["status"] == "undetermined"
    assert ok["status"] == "ok"
    # The undetermined entry's per-solve counter is its own.
    assert limited["attempts"] == 3
    assert ok["attempts"] == 2


def test_candidate_ok_when_baseline_hit_limit_reports_null_changes():
    packages = {
        "app": {"1.0": {"dependencies": {"p0": ""}}},
        "p0": {
            "1.0": {},
            "2.0": {"dependencies": {"p1": ""}},
        },
        "p1": {
            "1.0": {"dependencies": {"missing": ""}},
            "2.0": {"dependencies": {"missing": ""}},
        },
    }
    status, body = _feasibility(
        packages=packages,
        root={"app": ""},
        target="p0",
        candidates=["1.0"],
        max_attempts=2,
    )
    assert status == 200
    assert body["baseline"]["status"] == "limit"
    entry = body["candidates"][0]
    # The pin prunes the search enough to converge: still ok, just no
    # baseline solution to compute changes against.
    assert entry["status"] == "ok"
    assert entry["resolved"] == {"app": "1.0", "p0": "1.0"}
    assert entry["changes"] is None


def test_undetermined_is_never_a_conflict():
    packages = {
        "app": {"1.0": {"dependencies": {"p0": ""}}},
        "p0": {
            "1.0": {},
            "2.0": {"dependencies": {"p1": ""}},
        },
        "p1": {
            "1.0": {"dependencies": {"missing": ""}},
            "2.0": {"dependencies": {"missing": ""}},
        },
    }
    status, body = _feasibility(
        packages=packages,
        root={"app": ""},
        target="p0",
        candidates=["2.0"],
        max_attempts=1,
    )
    assert status == 200
    entry = body["candidates"][0]
    assert entry["status"] == "undetermined"
    assert entry["error"]["code"] == "search_limit_exceeded"
    # Hitting the limit still carries the diagnostic conflict structure.
    assert "conflict" in entry
    assert "resolved" not in entry
    assert body["summary"] == {
        "candidates": 1, "ok": 0, "conflict": 0, "undetermined": 1
    }


def test_other_locks_still_participate():
    # The lock is on a different package; candidate x==1.0 forces the closure
    # away from it because x 1.0 requires dep==1.0.  The lock disposition
    # follows the same rules as /resolve (the pinned dep 2.0 can never satisfy
    # ==1.0, so it is reported unsatisfiable rather than silently dropped).
    packages = {
        "x": {
            "1.0": {"dependencies": {"dep": "==1.0"}},
            "2.0": {"dependencies": {"dep": ">=2"}},
        },
        "dep": {"1.0": {}, "2.0": {}},
    }
    status, body = _feasibility(
        packages=packages,
        root=["x"],
        locks={"dep": "2.0"},
        target="x",
        candidates=["2.0", "1.0"],
    )
    assert status == 200
    assert body["candidates"][0]["locks"] == {"dep": "applied"}
    assert body["candidates"][0]["resolved"]["dep"] == "2.0"
    assert body["candidates"][1]["locks"] == {"dep": "unsatisfiable"}
    assert body["candidates"][1]["resolved"]["dep"] == "1.0"
    assert body["candidates"][1]["changes"]["changed"] == [
        {"package": "dep", "from": "2.0", "to": "1.0"},
        {"package": "x", "from": "2.0", "to": "1.0"},
    ]


def test_target_lock_still_participates():
    # A lock on the target itself remains active; the pin just contradicts it.
    status, body = _feasibility(
        packages={"x": {"1.0": {}, "2.0": {}}},
        root={"x": ""},
        locks={"x": "1.0"},
        target="x",
        candidates=["2.0", "1.0"],
    )
    assert status == 200
    upgraded, same = body["candidates"]
    assert upgraded["status"] == "ok"
    assert upgraded["locks"] == {"x": "unsatisfiable"}
    assert upgraded["resolved"]["x"] == "2.0"
    assert same["locks"] == {"x": "applied"}
    assert same["changes"]["changed"] == []


def test_prerelease_candidate_acts_as_an_explicit_prerelease_pin():
    packages = {"x": {"1.0": {}, "2.0rc1": {}}}
    status, body = _feasibility(
        packages=packages, root={"x": ""}, target="x", candidates=["2.0rc1"]
    )
    assert status == 200
    entry = body["candidates"][0]
    assert entry["status"] == "ok"
    assert entry["resolved"]["x"] == "2.0rc1"

    # allow_prereleases is honored for the baseline/other packages.
    assert body["baseline"]["resolved"] == {"x": "1.0"}
    status, body = _feasibility(
        packages=packages, root={"x": ""},
        target="x", candidates=["2.0rc1"], allow_prereleases=True,
    )
    assert body["baseline"]["resolved"]["x"] == "2.0rc1"


def test_marker_env_is_honored():
    packages = {
        "app": {"1.0": {"dependencies": ["dep>=2; python_version < '3.11'"]}},
        "dep": {"1.0": {}, "2.0": {}},
    }
    status, body = _feasibility(
        packages=packages,
        root=["app"],
        marker_env={"python_version": "3.11"},
        target="app",
        candidates=["1.0"],
    )
    assert status == 200
    assert body["candidates"][0]["resolved"] == {"app": "1.0"}


def test_target_and_candidates_are_normalized():
    packages = {"Foo_Bar": {"1.0.0": {}, "2.0": {}}}
    status, body = _feasibility(
        packages=packages,
        root={"foo-bar": ">=1.0"},
        target="FOO-BAR",
        candidates=["2.0", "1.0"],
    )
    assert status == 200
    assert body["target"]["package"] == "foo-bar"
    # Echoed with the index's own (normalized) version spelling.
    assert body["target"]["candidates"] == ["2.0", "1.0.0"]


def test_baseline_conflict_still_solves_every_candidate():
    packages = {
        "a": {"1.0": {"dependencies": {"x": ">=2.0"}}},
        "b": {"1.0": {"dependencies": {"x": "<2.0"}}},
        "x": {"1.0": {}, "2.0": {}},
    }
    status, body = _feasibility(
        packages=packages,
        root=["a", "b"],
        target="x",
        candidates=["1.0", "2.0"],
    )
    assert status == 200
    assert body["baseline"]["status"] == "conflict"
    # The added pin cannot repair an already contradictory root set.
    assert [entry["status"] for entry in body["candidates"]] == [
        "conflict", "conflict"
    ]
    # Conflicting entries carry no resolution or change baseline.
    for entry in body["candidates"]:
        assert "resolved" not in entry
        assert "changes" not in entry


def test_baseline_limit_is_reported_but_candidates_still_run():
    packages = {
        "app": {"1.0": {"dependencies": {"p0": ""}}},
        "p0": {
            "1.0": {},
            "2.0": {"dependencies": {"p1": ""}},
        },
        "p1": {
            "1.0": {"dependencies": {"missing": ""}},
            "2.0": {"dependencies": {"missing": ""}},
        },
    }
    status, body = _feasibility(
        packages=packages,
        root={"app": ""},
        target="p0",
        candidates=["1.0"],
        max_attempts=1,
    )
    assert status == 200
    assert body["baseline"]["status"] == "limit"
    assert body["baseline"]["error"]["code"] == "search_limit_exceeded"
    entry = body["candidates"][0]
    assert entry["status"] == "undetermined"
    assert "resolved" not in entry
    assert "changes" not in entry
    assert entry["error"]["code"] == "search_limit_exceeded"


def test_attempts_sum_baseline_and_all_candidates():
    status, body = _feasibility(
        packages={"x": {"1.0": {}, "2.0": {}}},
        root={"x": ""},
        target="x",
        candidates=["1.0", "2.0"],
    )
    assert status == 200
    # Baseline evaluates x==2.0 once; each pinned candidate evaluates once.
    assert body["baseline"]["attempts"] == 1
    assert [entry["attempts"] for entry in body["candidates"]] == [1, 1]
    assert body["attempts"] == 3


def test_deterministic_results():
    payload = {
        "packages": PACKAGES,
        "root": {"app": ""},
        "locks": {"web": "2.0", "common": "1.5"},
        "target": "web",
        "candidates": ["2.0", "3.0"],
    }
    results = [
        client.post("/analyze/upgrade-feasibility", json=payload).json()
        for _ in range(3)
    ]
    assert all(result == results[0] for result in results)


def test_supplied_index_is_not_mutated():
    # The pin lived only in the per-candidate solve; a follow-up /resolve
    # with the same index still picks the highest unconstrained version.
    payload = {
        "packages": PACKAGES,
        "root": {"app": ""},
        "locks": {"web": "2.0", "common": "1.5"},
        "target": "web",
        "candidates": ["2.0", "3.0"],
    }
    status, body = _feasibility(**payload)
    assert status == 200
    followup = client.post(
        "/resolve",
        json={
            "packages": PACKAGES,
            "root": {"app": ""},
            "locks": {"web": "2.0", "common": "1.5"},
        },
    ).json()
    assert followup["resolved"]["web"] == "2.0"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_unknown_target_package_is_an_input_error():
    status, body = _feasibility(
        packages={"x": {"1.0": {}}},
        root={"x": ""},
        target="ghost",
        candidates=["1.0"],
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"
    assert "ghost" in body["error"]["message"]


def test_unknown_candidate_version_is_an_input_error():
    status, body = _feasibility(
        packages={"x": {"1.0": {}}},
        root={"x": ""},
        target="x",
        candidates=["9.9"],
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"
    assert "9.9" in body["error"]["message"]
    assert "1.0" in body["error"]["message"]  # available versions listed


def test_invalid_candidate_version_is_an_input_error():
    status, body = _feasibility(
        packages={"x": {"1.0": {}}},
        root={"x": ""},
        target="x",
        candidates=["not-a-version"],
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"


def test_duplicate_candidates_after_pep440_normalization_are_rejected():
    status, body = _feasibility(
        packages={"x": {"1.0": {}, "2.0": {}}},
        root={"x": ""},
        target="x",
        candidates=["1.0", "1.0.0"],
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"
    assert "normalization" in body["error"]["message"]

    # Distinct normalized versions are fine.
    status, body = _feasibility(
        packages={"x": {"1.0": {}, "2.0": {}}},
        root={"x": ""},
        target="x",
        candidates=["1.0", "2.0"],
    )
    assert status == 200


def test_candidate_errors_are_collected_together():
    status, body = _feasibility(
        packages={"x": {"1.0": {}}},
        root={"x": ""},
        target="x",
        candidates=["nope", "9.9"],
    )
    assert status == 400
    message = body["error"]["message"]
    assert "nope" in message
    assert "9.9" in message
    assert ";" in message


def test_candidate_count_bounds():
    base = {
        "packages": {"x": {"1.0": {}}},
        "root": {"x": ""},
        "target": "x",
    }
    response = client.post(
        "/analyze/upgrade-feasibility", json={**base, "candidates": []}
    )
    assert response.status_code == 422
    response = client.post(
        "/analyze/upgrade-feasibility",
        json={**base, "candidates": ["1.0"] * 9},
    )
    assert response.status_code == 422


def test_shape_validation():
    base = {"packages": {"x": {"1.0": {}}}, "root": {"x": ""}}
    # Missing target/candidates.
    response = client.post("/analyze/upgrade-feasibility", json=base)
    assert response.status_code == 422
    # Empty target.
    response = client.post(
        "/analyze/upgrade-feasibility",
        json={**base, "target": "", "candidates": ["1.0"]},
    )
    assert response.status_code == 422
    # Unexpected top-level field.
    response = client.post(
        "/analyze/upgrade-feasibility",
        json={**base, "target": "x", "candidates": ["1.0"], "extra": 1},
    )
    assert response.status_code == 422


def test_resolve_input_validation_is_reused():
    common = {"target": "x", "candidates": ["1.0"]}
    status, body = _feasibility(
        packages={"x": {"not-a-version": {}}}, root={"x": ""}, **common
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"

    status, body = _feasibility(
        packages={"x": {"1.0": {}}}, root=">>>1", **common
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"

    status, body = _feasibility(
        packages={"x": {"1.0": {}}}, root={"x": ""},
        locks={"x": "not-a-version"}, **common,
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"
