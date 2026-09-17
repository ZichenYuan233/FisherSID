"""Deterministic multi-level residual k-means."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


def _squared_distances(x: Tensor, centers: Tensor) -> Tensor:
    return (x.square().sum(1, keepdim=True) + centers.square().sum(1) - 2 * x @ centers.T).clamp_min_(0)


def _kmeans_plus_plus(x: Tensor, k: int, generator: torch.Generator) -> Tensor:
    n = x.shape[0]
    first = torch.randint(n, (1,), generator=generator, device=x.device).item()
    chosen = [x[first]]
    closest = (x - chosen[0]).square().sum(1)
    for _ in range(1, k):
        total = closest.sum()
        if total <= 0:
            index = torch.randint(n, (1,), generator=generator, device=x.device).item()
        else:
            index = torch.multinomial(closest / total, 1, generator=generator).item()
        chosen.append(x[index])
        closest = torch.minimum(closest, (x - x[index]).square().sum(1))
    return torch.stack(chosen)


def _fit_kmeans(x: Tensor, k: int, max_iters: int, seed: int, tol: float) -> Tensor:
    k = min(k, x.shape[0])
    generator = torch.Generator(device=x.device).manual_seed(seed)
    centers = _kmeans_plus_plus(x, k, generator)
    previous_loss = None
    for _ in range(max_iters):
        distances = _squared_distances(x, centers)
        labels = distances.argmin(1)
        counts = torch.bincount(labels, minlength=k)
        new_centers = torch.zeros_like(centers)
        new_centers.index_add_(0, labels, x)
        nonempty = counts > 0
        new_centers[nonempty] /= counts[nonempty, None]
        if (~nonempty).any():
            farthest = distances.min(1).values.argsort(descending=True)
            new_centers[~nonempty] = x[farthest[: int((~nonempty).sum())]]
        loss = distances.gather(1, labels[:, None]).mean()
        centers = new_centers
        if previous_loss is not None and abs(previous_loss - loss.item()) <= tol * max(previous_loss, 1.0):
            break
        previous_loss = loss.item()
    return centers


@dataclass
class ResidualKMeans:
    levels: int = 4
    codebook_size: int = 256
    max_iters: int = 100
    seed: int = 0
    tol: float = 1e-5
    codebooks_: list[Tensor] = field(default_factory=list)

    def fit(self, x: Tensor) -> "ResidualKMeans":
        residual = x.detach().clone()
        self.codebooks_ = []
        for level in range(self.levels):
            centers = _fit_kmeans(
                residual, self.codebook_size, self.max_iters, self.seed + level, self.tol
            )
            self.codebooks_.append(centers)
            labels = _squared_distances(residual, centers).argmin(1)
            residual = residual - centers[labels]
        return self

    def encode(self, x: Tensor) -> Tensor:
        residual = x
        codes = []
        for centers in self.codebooks_:
            labels = _squared_distances(residual, centers).argmin(1)
            codes.append(labels)
            residual = residual - centers[labels]
        return torch.stack(codes, dim=1)

    def decode(self, codes: Tensor) -> Tensor:
        reconstruction = torch.zeros(
            codes.shape[0], self.codebooks_[0].shape[1],
            dtype=self.codebooks_[0].dtype, device=self.codebooks_[0].device,
        )
        for level, centers in enumerate(self.codebooks_):
            reconstruction += centers[codes[:, level]]
        return reconstruction

    def fit_encode(self, x: Tensor) -> Tensor:
        return self.fit(x).encode(x)
