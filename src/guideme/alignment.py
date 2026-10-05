"""Numerical alignment of an execution window to a reference subsequence."""

import numpy as np


def subsequence_dtw(cost, *, warp_penalty: float = 0.0) -> dict:
    """Match every execution row to a contiguous reference subsequence.

    Accept a nonempty, finite, nonnegative execution-by-reference cost matrix.
    Reference start and end are free; execution rows are all consumed. This
    matches a supplied window, not an execution prefix or a semantic step.
    Predecessor ties prefer diagonal, up, then left. Endpoint ties choose the
    earliest reference index while exposing all tied endpoints to the caller.
    A finite nonnegative warp_penalty adds cost to each up or left move,
    discouraging repeated frames without imposing a speed limit. Diagonal
    moves and the first matched cell incur no penalty; tutorial boundaries
    remain free. Zero preserves the original matching behavior. All returned
    costs include the penalty. total_cost retains the sum; normalized_cost
    and endpoint_costs divide by execution row count, not path length.
    Invalid inputs or accumulated float64 overflow raise ValueError.
    The caller records the chosen penalty.
    """
    if isinstance(warp_penalty, bool) or not isinstance(warp_penalty, (int, float)):
        raise ValueError("warp_penalty must be a finite nonnegative number")
    try:
        warp_penalty = float(warp_penalty)
    except OverflowError as error:
        raise ValueError("warp_penalty must be a finite nonnegative number") from error
    if not np.isfinite(warp_penalty) or warp_penalty < 0:
        raise ValueError("warp_penalty must be a finite nonnegative number")

    try:
        costs = np.asarray(cost, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("cost must be a numeric matrix") from error
    if (
        costs.ndim != 2
        or 0 in costs.shape
        or not np.isfinite(costs).all()
        or (costs < 0).any()
    ):
        raise ValueError("cost must be a nonempty finite nonnegative 2D matrix")

    execution_count, reference_count = costs.shape
    accumulated = np.full((execution_count + 1, reference_count + 1), np.inf)
    accumulated[0, :] = 0.0
    predecessors = np.empty(costs.shape, dtype=np.uint8)
    try:
        with np.errstate(over="raise", invalid="raise"):
            for i in range(1, execution_count + 1):
                for j in range(1, reference_count + 1):
                    candidates = (
                        accumulated[i - 1, j - 1],
                        # An overflowing alternative is unreachable; reject
                        # overflow only if the chosen accumulated value is not finite.
                        float(accumulated[i - 1, j]) + warp_penalty,
                        float(accumulated[i, j - 1]) + warp_penalty,
                    )
                    direction = int(np.argmin(candidates))
                    predecessors[i - 1, j - 1] = direction
                    accumulated[i, j] = costs[i - 1, j - 1] + candidates[direction]
                    if not np.isfinite(accumulated[i, j]):
                        raise ValueError("accumulated cost exceeds finite float64 range")
    except FloatingPointError as error:
        raise ValueError("accumulated cost exceeds finite float64 range") from error

    endpoint_totals = accumulated[-1, 1:]
    total_cost = float(endpoint_totals.min())
    tied_endpoints = np.flatnonzero(endpoint_totals == total_cost).tolist()
    end_reference_index = tied_endpoints[0]
    i, j = execution_count, end_reference_index + 1
    path = []
    while i > 0:
        path.append((i - 1, j - 1))
        direction = predecessors[i - 1, j - 1]
        if direction == 0:
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()

    return {
        "path": path,
        "start_reference_index": path[0][1],
        "end_reference_index": end_reference_index,
        "total_cost": total_cost,
        "normalized_cost": total_cost / execution_count,
        "endpoint_costs": (endpoint_totals / execution_count).tolist(),
        "tied_endpoints": tied_endpoints,
    }


def align_feature_window(reference_features, execution_features, *, warp_penalty: float = 0.0) -> dict:
    """Align an execution feature window using cosine distance and DTW.

    Require nonempty finite real matrices with the same feature dimension and
    no zero rows, whose cosine direction would be undefined. Normalize copies
    using row scaling before L2 norms to handle very large or small values.
    Return the numerical subsequence_dtw result without step/error semantics.
    Forward warp_penalty to that matcher for validation and scoring; zero
    preserves existing behavior. Returned costs include any warp penalty.
    """
    normalized = []
    for name, features in (
        ("reference_features", reference_features),
        ("execution_features", execution_features),
    ):
        try:
            values = np.asarray(features)
            if np.iscomplexobj(values):
                raise ValueError(f"{name} must contain real values")
            values = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a real numeric matrix") from error
        if values.ndim != 2 or 0 in values.shape or not np.isfinite(values).all():
            raise ValueError(f"{name} must be a nonempty finite 2D matrix")
        row_scale = np.max(np.abs(values), axis=1, keepdims=True)
        if (row_scale == 0).any():
            raise ValueError(f"{name} must not contain zero rows")
        scaled = values / row_scale
        normalized.append(scaled / np.linalg.norm(scaled, axis=1, keepdims=True))

    reference_unit, execution_unit = normalized
    if reference_unit.shape[1] != execution_unit.shape[1]:
        raise ValueError("reference and execution feature dimensions must match")
    costs = np.clip(1.0 - execution_unit @ reference_unit.T, 0.0, 2.0)
    return subsequence_dtw(costs, warp_penalty=warp_penalty)
