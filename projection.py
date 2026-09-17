"""Fisher direction selection and sensitivity weighting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .fisher import fisher_eigenspace


@dataclass
class FisherProjector:
    rank: int = 64
    rho: float = 0.5
    relative_floor: float = 1e-6
    mean_: Tensor | None = None
    eigenvalues_: Tensor | None = None
    components_: Tensor | None = None
    weights_: Tensor | None = None

    def fit(self, item_features: Tensor, fisher: Tensor) -> "FisherProjector":
        space = fisher_eigenspace(
            fisher, rank=self.rank, relative_floor=self.relative_floor
        )
        self.mean_ = item_features.mean(dim=0)
        self.eigenvalues_ = space.eigenvalues.to(item_features)
        self.components_ = space.eigenvectors.to(item_features)
        normalized = self.eigenvalues_ / self.eigenvalues_[0]
        self.weights_ = normalized.pow(self.rho)
        return self

    def _check_fitted(self) -> None:
        return None

    def transform(self, item_features: Tensor) -> Tensor:
        self._check_fitted()
        return ((item_features - self.mean_) @ self.components_) * self.weights_

    def inverse_transform(self, transformed: Tensor) -> Tensor:
        """Map weighted coordinates back to the retained feature subspace."""
        self._check_fitted()
        return (transformed / self.weights_) @ self.components_.T + self.mean_

    def projection_reconstruction(self, item_features: Tensor) -> Tensor:
        """Equation (11): unweighted reconstruction for projection evaluation."""
        self._check_fitted()
        centered = item_features - self.mean_
        return centered @ self.components_ @ self.components_.T + self.mean_
