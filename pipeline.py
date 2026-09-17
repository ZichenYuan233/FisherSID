"""End-to-end FisherSID tokenizer."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from .projection import FisherProjector
from .rq import ResidualKMeans


@dataclass
class FisherSID:
    rank: int = 64
    rho: float = 0.5
    levels: int = 4
    codebook_size: int = 256
    max_iters: int = 100
    seed: int = 0
    projector: FisherProjector = field(init=False)
    quantizer: ResidualKMeans = field(init=False)

    def __post_init__(self) -> None:
        self.projector = FisherProjector(rank=self.rank, rho=self.rho)
        self.quantizer = ResidualKMeans(
            levels=self.levels,
            codebook_size=self.codebook_size,
            max_iters=self.max_iters,
            seed=self.seed,
        )

    def fit(self, item_features: Tensor, fisher: Tensor) -> "FisherSID":
        z = self.projector.fit(item_features, fisher).transform(item_features)
        self.quantizer.fit(z)
        return self

    def encode(self, item_features: Tensor) -> Tensor:
        return self.quantizer.encode(self.projector.transform(item_features))

    def reconstruct_features(self, codes: Tensor) -> Tensor:
        return self.projector.inverse_transform(self.quantizer.decode(codes))

    def state_dict(self) -> dict:
        self.projector._check_fitted()
        return {
            "config": {
                "rank": self.rank, "rho": self.rho, "levels": self.levels,
                "codebook_size": self.codebook_size, "max_iters": self.max_iters,
                "seed": self.seed,
            },
            "mean": self.projector.mean_,
            "eigenvalues": self.projector.eigenvalues_,
            "components": self.projector.components_,
            "weights": self.projector.weights_,
            "codebooks": self.quantizer.codebooks_,
        }

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, *, map_location: str | torch.device = "cpu") -> "FisherSID":
        state = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**state["config"])
        model.projector.mean_ = state["mean"]
        model.projector.eigenvalues_ = state["eigenvalues"]
        model.projector.components_ = state["components"]
        model.projector.weights_ = state["weights"]
        model.quantizer.codebooks_ = state["codebooks"]
        return model
