"""Tokenizers used by the fixed-budget baseline sweep."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn

from .pipeline import FisherSID
from .rq import ResidualKMeans


@dataclass
class LinearRQTokenizer:
    method: str
    rank: int = 64
    levels: int = 4
    codebook_size: int = 256
    max_iters: int = 100
    seed: int = 0
    mean_: Tensor | None = None
    components_: Tensor | None = None
    rq_: ResidualKMeans | None = None

    def fit(self, features: Tensor, *, projection: Tensor | None = None) -> "LinearRQTokenizer":
        self.mean_ = features.mean(0)
        centered = features - self.mean_
        if self.method == "raw_rq":
            z = centered
            self.components_ = torch.eye(features.shape[1], device=features.device, dtype=features.dtype)
        elif self.method == "pca_rq":
            _, _, vectors = torch.pca_lowrank(centered, q=min(self.rank, centered.shape[1]))
            self.components_ = vectors[:, : self.rank]
            z = centered @ self.components_
        elif self.method in {"affinity_projection", "teacher_projection"}:
            self.components_ = projection[:, : self.rank].to(features)
            z = centered @ self.components_
        else:
            self.components_ = torch.eye(features.shape[1], device=features.device, dtype=features.dtype)
            z = centered
        self.rq_ = ResidualKMeans(self.levels, self.codebook_size, self.max_iters, self.seed).fit(z)
        return self

    def encode(self, features: Tensor) -> Tensor:
        return self.rq_.encode((features - self.mean_) @ self.components_)


class RQVAE(nn.Module):
    """Shared RQ-VAE backbone for adapted neural SID baselines."""

    def __init__(self, input_dim: int, latent_dim: int = 64, levels: int = 4, codebook_size: int = 256):
        super().__init__()
        self.levels = levels
        self.codebook_size = codebook_size
        self.encoder = nn.Sequential(nn.Linear(input_dim, 512), nn.GELU(), nn.Linear(512, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 512), nn.GELU(), nn.Linear(512, input_dim))
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, latent_dim) / latent_dim**0.5) for _ in range(levels)
        ])

    def quantize(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        residual = z
        quantized = torch.zeros_like(z)
        codes = []
        commitment = z.new_tensor(0.0)
        for codebook in self.codebooks:
            distances = residual.square().sum(1, keepdim=True) + codebook.square().sum(1) - 2 * residual @ codebook.T
            index = distances.argmin(1)
            vector = codebook[index]
            codes.append(index)
            commitment = commitment + nn.functional.mse_loss(residual, vector.detach()) + 0.25 * nn.functional.mse_loss(vector, residual.detach())
            quantized = quantized + vector
            residual = residual - vector
        straight_through = z + (quantized - z).detach()
        return straight_through, torch.stack(codes, 1), commitment

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        z = self.encoder(x)
        quantized, codes, commitment = self.quantize(z)
        return self.decoder(quantized), z, codes, commitment


def train_neural_baseline(
    method: str,
    features: Tensor,
    *,
    auxiliary: Tensor | None = None,
    latent_dim: int = 64,
    levels: int = 4,
    codebook_size: int = 256,
    epochs: int = 100,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    device: str = "cuda",
) -> tuple[RQVAE, Tensor]:
    """Train TIGER/LETTER/MME-SID/MusicRec/PIT adapted tokenizers.

    ``auxiliary`` supplies collaborative, fused-modal, graph, or teacher target
    representations for LETTER, MME-SID, MusicRec, and PIT respectively.
    """
    model = RQVAE(features.shape[1], latent_dim, levels, codebook_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    permutation_generator = torch.Generator().manual_seed(0)
    for _ in range(epochs):
        permutation = torch.randperm(len(features), generator=permutation_generator)
        for start in range(0, len(features), batch_size):
            ids = permutation[start : start + batch_size]
            x = features[ids].to(device)
            reconstruction, latent, _, commitment = model(x)
            loss = nn.functional.mse_loss(reconstruction, x) + commitment
            if auxiliary is not None and method != "tiger":
                target = auxiliary[ids].to(device)
                width = min(latent.shape[1], target.shape[1])
                loss = loss + 0.1 * nn.functional.mse_loss(latent[:, :width], target[:, :width])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        _, _, codes, _ = model(features.to(device))
    return model.cpu(), codes.cpu()


def make_tokenizer(method: str, features: Tensor, fisher: Tensor | None, **kwargs):
    if method == "fishersid":
        return FisherSID(rho=0.5, **kwargs).fit(features, fisher)
    if method == "fisher_rq":
        return FisherSID(rho=0.0, **kwargs).fit(features, fisher)
    tokenizer = LinearRQTokenizer(method=method, **kwargs)
    return tokenizer.fit(features)
