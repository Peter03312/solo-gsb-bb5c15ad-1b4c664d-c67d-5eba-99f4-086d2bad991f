"""Upgrade-feasibility analysis.

Given the same package index, root requirements, locks and solve options as
``/resolve`` plus one target package and 1-8 candidate versions, this module
answers, *for every candidate in input order*: *can the whole dependency
closure still be resolved if the target is pinned to exactly that version?*

Semantics
---------
* A baseline solve first resolves the unmodified input — the "current state"
  under the supplied root requirements and locks (locks keep their
  lock-first behaviour).  It is the comparison baseline for the version
  changes reported by every successful candidate.
* Each candidate is evaluated by an **independent, exhaustive solve** on the
  same index with one extra *root* constraint ``target==candidate``
  temporarily added.  The original root requirements and **all** original
  locks still participate, so feasibility is checked against the complete
  dependency constraints — the target package's own version range is never
  checked in isolation.  Nothing about one candidate's search affects
  another, and the supplied index is never mutated.
* Candidates keep their input order.  Each entry reports ``ok``,
  ``conflict`` or ``undetermined`` and its own ``attempts`` count:

  - ``ok`` — the full resolution, lock disposition and the version changes
    relative to the baseline solution;
  - ``conflict`` — the structured conflict summary;
  - ``undetermined`` — the solve hit ``max_attempts``; hitting the search
    limit is not treated as unsatisfiable.

* Candidate versions must exist in the index for the target package and may
  not repeat after PEP 440 normalization; either violation is a 400 input
  error for the whole request.
"""

from __future__ import annotations

from typing import Any, Optional

from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from .resolver import (
    PackageIndex,
    ResolutionInputError,
    RootEntry,
    SolveResult,
    parse_requirements,
    resolve,
)
from .serialize import conflict_summary


def parse_upgrade_target(
    index: PackageIndex, package: str, raw_candidates: list[str]
) -> tuple[str, list[Version]]:
    """Validate and normalize the target package and candidate versions.

    Returns the canonical package name and the parsed candidate versions in
    input order (each the matched index ``Version`` object, whose string
    form is the index's own normalized spelling).  The package must exist in
    the index; every candidate must parse as PEP 440, exist among the
    target's index versions, and be unique after PEP 440 normalization
    (``1.0`` and ``1.0.0`` are the same candidate).  All violations are
    reported together as one 400 error.
    """
    canonical = canonicalize_name(package)
    if canonical not in index.versions:
        raise ResolutionInputError(
            f"target: package {package!r} is not present in the supplied index"
        )

    index_versions = [version for version, _record in index.versions[canonical]]
    errors: list[str] = []
    candidates: list[Version] = []
    seen: list[tuple[str, Version]] = []
    for raw in raw_candidates:
        try:
            parsed = Version(raw)
        except InvalidVersion as exc:
            errors.append(
                f"target candidates: {raw!r} is not a valid PEP 440 version: {exc}"
            )
            continue
        # PEP 440 equality: 1.0 and 1.0.0 normalize to the same version even
        # though their spellings differ (matching is Version-equality based,
        # exactly like the rest of the solver).
        duplicate = next((first for first, version in seen if version == parsed), None)
        if duplicate is not None:
            errors.append(
                f"target candidates: {raw!r} duplicates {duplicate!r} "
                "after PEP 440 normalization"
            )
            continue
        matched = next(
            (version for version in index_versions if version == parsed), None
        )
        if matched is None:
            offered = ", ".join(
                str(version) for version in sorted(index_versions, reverse=True)
            )
            errors.append(
                f"target candidates: package {canonical!r} has no version "
                f"{raw!r} in the supplied index (available: {offered or 'none'})"
            )
            continue
        seen.append((raw, matched))
        candidates.append(matched)

    if errors:
        raise ResolutionInputError("; ".join(errors))
    return canonical, candidates


def _changes(baseline: dict[str, str], candidate: dict[str, str]) -> dict[str, Any]:
    """Diff two successful resolutions; every list is sorted by package name."""
    added = [
        {"package": name, "version": candidate[name]}
        for name in sorted(candidate.keys() - baseline.keys())
    ]
    removed = [
        {"package": name, "version": baseline[name]}
        for name in sorted(baseline.keys() - candidate.keys())
    ]
    changed = [
        {"package": name, "from": baseline[name], "to": candidate[name]}
        for name in sorted(baseline.keys() & candidate.keys())
        if baseline[name] != candidate[name]
    ]
    return {"added": added, "removed": removed, "changed": changed}


def _baseline_body(result: SolveResult, max_attempts: int) -> dict[str, Any]:
    """Render the baseline outcome in the ``/resolve`` body shape."""
    body: dict[str, Any] = {
        "status": result.status,
        "attempts": result.attempts,
    }
    if result.status == "ok":
        body["resolved"] = result.chosen
        body["dependency_graph"] = result.edges
        body["locks"] = result.lock_notes
        return body
    body["locks"] = result.lock_notes
    body["conflict"] = conflict_summary(result)
    if result.status == "limit":
        body["error"] = {
            "code": "search_limit_exceeded",
            "message": (
                f"baseline solve exceeded {max_attempts} candidate evaluations; "
                "the unmodified input may still be satisfiable"
            ),
        }
    return body


def _candidate_body(
    version: Version,
    result: SolveResult,
    max_attempts: int,
    baseline: Optional[dict[str, str]],
) -> dict[str, Any]:
    """Render one candidate's independent solve as a public entry."""
    body: dict[str, Any] = {
        "version": str(version),
        "status": "ok" if result.status == "ok" else (
            "undetermined" if result.status == "limit" else "conflict"
        ),
        "attempts": result.attempts,
    }
    if result.status == "ok":
        body["resolved"] = result.chosen
        body["dependency_graph"] = result.edges
        body["locks"] = result.lock_notes
        # A candidate solution only needs a comparison baseline.  When the
        # baseline hit the search limit there is no baseline solution to diff
        # (the pin may simply have pruned the search); the candidate is still
        # ok, so changes is explicitly null rather than fabricated.
        body["changes"] = _changes(baseline, result.chosen) if baseline is not None else None
        return body
    body["locks"] = result.lock_notes
    body["conflict"] = conflict_summary(result)
    if result.status == "limit":
        body["error"] = {
            "code": "search_limit_exceeded",
            "message": (
                f"resolution with {version} pinned exceeded {max_attempts} "
                "candidate evaluations; the candidate may still be feasible"
            ),
        }
    return body


def run_upgrade_feasibility(
    index: PackageIndex,
    entries: list[RootEntry],
    locks: dict[str, Version],
    *,
    allow_prereleases: bool,
    marker_env: Optional[dict[str, str]],
    max_attempts: int,
    target_package: str,
    raw_candidates: list[str],
) -> dict[str, Any]:
    """Solve the baseline and every pinned candidate; render the JSON body."""
    target, candidates = parse_upgrade_target(index, target_package, raw_candidates)
    roots = [entry.dep for entry in entries]

    def solve(extra_roots: list) -> SolveResult:
        return resolve(
            index,
            roots + extra_roots,
            locks,
            allow_prereleases=allow_prereleases,
            marker_env=marker_env,
            max_attempts=max_attempts,
        )

    baseline = solve([])
    total_attempts = baseline.attempts
    baseline_chosen = baseline.chosen if baseline.status == "ok" else None

    bodies: list[dict[str, Any]] = []
    counts = {"ok": 0, "conflict": 0, "undetermined": 0}
    for version in candidates:
        # An explicit root pin such as ``pkg==2.0rc1`` goes through the same
        # PEP 508 parsing as every other root requirement (pre-release pins
        # relax pre-release exclusion for the target package only).
        pin = parse_requirements(
            [f"{target}=={version}"],
            origin=f"upgrade feasibility candidate {str(version)!r}",
        )[0]
        result = solve([pin])
        total_attempts += result.attempts
        body = _candidate_body(version, result, max_attempts, baseline_chosen)
        counts[body["status"]] += 1
        bodies.append(body)

    return {
        "target": {
            "package": target,
            "candidates": [str(version) for version in candidates],
        },
        "allow_prereleases": allow_prereleases,
        "summary": {
            "candidates": len(candidates),
            "ok": counts["ok"],
            "conflict": counts["conflict"],
            "undetermined": counts["undetermined"],
        },
        "baseline": _baseline_body(baseline, max_attempts),
        "candidates": bodies,
        "attempts": total_attempts,
    }
