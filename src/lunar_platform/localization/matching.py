"""
Crater constellation matching.

Each observed candidate is known only as an offset from the rover. Pairing a
candidate with a reference crater therefore implies a rover position:

    implied_rover = reference_position - candidate_offset

If a pairing set is correct, every pair implies the same rover position, so the
correct solution shows up as the densest cluster in translation space. That is
what this does - a Hough-style vote over implied positions, gated by relative
crater size, with the winning cluster refined by averaging its members.

MVP assumption, stated rather than hidden: the rover heading is taken as known.
Solving heading too means voting over rotation as well, which needs a
rotation-invariant descriptor (inter-crater distance signatures). The interface
below returns rotation explicitly so that extension does not change callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..crater_mapping.models import CraterCandidate, CraterMatch, ReferenceCrater


@dataclass
class MatchSolution:
    estimated_x_m: float | None
    estimated_y_m: float | None
    rotation_deg: float
    matches: list[CraterMatch] = field(default_factory=list)
    inlier_ratio: float = 0.0
    n_votes: int = 0


def _cluster_spread_m(
    vote_xy: np.ndarray, candidate_idx: np.ndarray, within: np.ndarray, seed: int
) -> float:
    """
    How tightly a cluster's votes actually agree, in metres RMS.

    One vote per distinct candidate - its closest to the seed - so a single
    candidate matching several look-alike landmarks cannot make a loose cluster
    look tight.
    """
    members = np.flatnonzero(within[seed])
    if len(members) == 0:
        return float("inf")

    closest: dict[int, tuple[float, int]] = {}
    for member in members:
        distance = float(
            np.hypot(
                vote_xy[member, 0] - vote_xy[seed, 0],
                vote_xy[member, 1] - vote_xy[seed, 1],
            )
        )
        key = int(candidate_idx[member])
        if key not in closest or distance < closest[key][0]:
            closest[key] = (distance, int(member))

    points = vote_xy[[index for _, index in closest.values()]]
    centre = points.mean(axis=0)
    return float(np.sqrt(((points - centre) ** 2).sum(axis=1).mean()))


def match_constellation(
    candidates: list[CraterCandidate],
    references: list[ReferenceCrater],
    position_tolerance_m: float = 18.0,
    diameter_tolerance: float = 0.35,
    min_inliers: int = 4,
    ambiguity_ratio: float = 0.75,
    ambiguity_spread_ratio: float = 3.0,
) -> MatchSolution:
    """
    Recover the rover position implied by the observed crater constellation.

    position_tolerance_m sets how far two implied positions may differ and
    still be treated as the same vote; it should exceed the aggregation
    centroid error but stay well below typical crater spacing.
    """
    if not candidates or not references:
        return MatchSolution(None, None, 0.0)

    # Every plausible pairing casts one vote for a rover position.
    votes: list[tuple[float, float, int, int]] = []
    for ci, candidate in enumerate(candidates):
        for ri, reference in enumerate(references):
            if reference.diameter_m <= 0:
                continue
            size_error = abs(candidate.diameter_m - reference.diameter_m) / reference.diameter_m
            if size_error > diameter_tolerance:
                continue
            votes.append(
                (
                    reference.x_m - candidate.offset_dx_m,
                    reference.y_m - candidate.offset_dy_m,
                    ci,
                    ri,
                )
            )

    if not votes:
        return MatchSolution(None, None, 0.0)

    vote_xy = np.array([[v[0], v[1]] for v in votes])

    # Score each vote by how many DISTINCT candidates agree with it. Counting
    # raw pairings instead would let one candidate matching several look-alike
    # landmarks outscore a genuine cluster of several different candidates.
    candidate_idx = np.array([v[2] for v in votes])
    deltas = vote_xy[:, None, :] - vote_xy[None, :, :]
    distances = np.hypot(deltas[:, :, 0], deltas[:, :, 1])
    within = distances <= position_tolerance_m
    support = np.array(
        [len(np.unique(candidate_idx[within[i]])) for i in range(len(votes))]
    )

    best_idx = int(np.argmax(support))
    best_support = int(support[best_idx])
    seed_x, seed_y = vote_xy[best_idx]

    if best_support < min_inliers:
        return MatchSolution(None, None, 0.0, [], 0.0, len(votes))

    # A rival cluster of comparable strength elsewhere means the constellation
    # is not distinctive enough to place. Reporting no fix is the honest
    # outcome; picking the marginal winner produces confident wrong positions.
    #
    # Strength alone is not enough to call something a rival, though. Vote
    # counts sit on a background of coincidental agreement - a handful of
    # candidates will always find some landmarks loosely arranged within the
    # tolerance ball - and that background does not shrink when the winner
    # does. Judging on counts alone therefore rejects correct fixes precisely
    # when few craters are visible, which is when a fix is most wanted.
    #
    # Measured on this demo map, a true cluster's votes agree to about 1-2 m
    # while a coincidental one scatters over 7-11 m inside an 18 m tolerance.
    # So a rival has to be tight as well as populous before it counts as one.
    far = distances[best_idx] > 3.0 * position_tolerance_m
    if np.any(far):
        far_indices = np.flatnonzero(far)
        rival_idx = int(far_indices[np.argmax(support[far])])
        runner_up = int(support[rival_idx])
        if runner_up >= ambiguity_ratio * best_support:
            best_spread = _cluster_spread_m(vote_xy, candidate_idx, within, best_idx)
            rival_spread = _cluster_spread_m(vote_xy, candidate_idx, within, rival_idx)
            credible_rival = rival_spread <= max(
                ambiguity_spread_ratio * best_spread, 1e-6
            )
            if credible_rival:
                return MatchSolution(None, None, 0.0, [], 0.0, len(votes))
    agreeing = np.flatnonzero(
        np.hypot(vote_xy[:, 0] - seed_x, vote_xy[:, 1] - seed_y) <= position_tolerance_m
    )

    # One candidate may agree via several references; keep its closest pairing.
    best_per_candidate: dict[int, tuple[float, int, int]] = {}
    for vi in agreeing:
        x, y, ci, ri = votes[vi]
        distance = float(np.hypot(x - seed_x, y - seed_y))
        if ci not in best_per_candidate or distance < best_per_candidate[ci][0]:
            best_per_candidate[ci] = (distance, ci, ri)

    if len(best_per_candidate) < min_inliers:
        return MatchSolution(None, None, 0.0, [], 0.0, int(best_support))

    inlier_positions = np.array(
        [
            [
                references[ri].x_m - candidates[ci].offset_dx_m,
                references[ri].y_m - candidates[ci].offset_dy_m,
            ]
            for _, ci, ri in best_per_candidate.values()
        ]
    )
    estimated = inlier_positions.mean(axis=0)

    matches = []
    for _, ci, ri in sorted(best_per_candidate.values(), key=lambda t: candidates[t[1]].id):
        candidate, reference = candidates[ci], references[ri]
        predicted_x = estimated[0] + candidate.offset_dx_m
        predicted_y = estimated[1] + candidate.offset_dy_m
        residual = float(np.hypot(predicted_x - reference.x_m, predicted_y - reference.y_m))
        matches.append(
            CraterMatch(
                candidate_id=candidate.id,
                reference_id=reference.id,
                residual_m=round(residual, 3),
            )
        )

    return MatchSolution(
        estimated_x_m=float(estimated[0]),
        estimated_y_m=float(estimated[1]),
        rotation_deg=0.0,
        matches=matches,
        inlier_ratio=float(len(matches) / len(candidates)),
        n_votes=len(votes),
    )
