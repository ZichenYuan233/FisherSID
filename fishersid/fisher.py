"""Ranking-Fisher estimation for item-separable teachers.

The implementation follows equations (5)--(7) in the paper.  A teacher is
item-separable when candidate feature ``h_i`` affects only logit ``i``.  Under
that assumption, differentiating the sum of logits returns every per-item
score gradient in one backward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Tuple

import torch
from torch import Tensor


ScoreFn = Callable[[Any, Tensor], Tensor]
FisherBatch = Tuple[Any, Tensor]


@dataclass(frozen=True)
class FisherEigenspace:
    """Leading eigenpairs of the shared ranking Fisher matrix."""

    eigenvalues: Tensor
    eigenvectors: Tensor


def estimate_ranking_fisher(
    score_fn: ScoreFn,
    batches: Iterable[FisherBatch],
    *,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Estimate the shared ranking Fisher without materializing ``G``.

    Args:
        score_fn: ``score_fn(user_context, candidate_features) -> [B, C]``.
            It must be differentiable and item-separable in its second input.
        batches: Pairs of user context and candidate features shaped ``[B,C,D]``.
        dtype: Accumulation dtype. Float64 is the safe default for eigendecomp.

    Returns:
        ``[D,D]`` matrix equal to the sample average over candidate sets of
        ``sum_i p_i(1-p_i) a_i a_i^T``.
    """
    fisher: Tensor | None = None
    num_sets = 0

    for user_context, candidate_features in batches:
        features = candidate_features.detach().requires_grad_(True)
        logits = score_fn(user_context, features)

        # For an item-separable teacher, d(sum_j logit_j)/d(h_i) = d(logit_i)/d(h_i).
        (score_grad,) = torch.autograd.grad(logits.sum(), features, create_graph=False)
        probs = logits.detach().softmax(dim=-1)
        weights = (probs * (1.0 - probs)).clamp_min(0).sqrt().unsqueeze(-1)
        weighted_grad = (weights * score_grad.detach()).reshape(-1, features.shape[-1])
        gram = weighted_grad.to(dtype).T @ weighted_grad.to(dtype)
        fisher = gram if fisher is None else fisher + gram
        num_sets += features.shape[0]

    fisher = fisher / num_sets
    return 0.5 * (fisher + fisher.T)


def fisher_eigenspace(
    fisher: Tensor,
    *,
    rank: int,
    relative_floor: float = 1e-6,
) -> FisherEigenspace:
    """Return descending leading eigenpairs, applying the paper's cutoff."""
    values, vectors = torch.linalg.eigh(0.5 * (fisher + fisher.T))
    order = torch.argsort(values, descending=True)
    values = values[order].clamp_min(0)
    vectors = vectors[:, order]
    keep = values / values[0] >= relative_floor
    effective_rank = min(rank, int(keep.sum().item()))
    return FisherEigenspace(values[:effective_rank], vectors[:, :effective_rank])
