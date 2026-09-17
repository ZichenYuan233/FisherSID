"""Teacher and autoregressive Semantic-ID recommender."""

from __future__ import annotations

import math
import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_length: int = 4096):
        super().__init__()
        position = torch.arange(max_length).float()[:, None]
        frequency = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        encoding = torch.zeros(max_length, dim)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.encoding[: x.shape[1]].to(x)


class RankingTeacher(nn.Module):
    """Two-layer history Transformer with an item-separable scoring head."""

    def __init__(
        self,
        feature_dim: int = 1536,
        hidden_dim: int = 256,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        max_history: int = 50,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.item_input = nn.Linear(feature_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim, heads, hidden_dim * 4, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.position = PositionalEncoding(hidden_dim, max_history + 1)
        self.user_norm = nn.LayerNorm(hidden_dim)
        self.item_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def encode_user(self, history_features: Tensor, history_mask: Tensor) -> Tensor:
        x = self.position(self.item_input(history_features))
        x = self.encoder(x, src_key_padding_mask=~history_mask)
        user = x[:, -1]
        return self.user_norm(user)

    def score(self, user_vectors: Tensor, candidate_features: Tensor) -> Tensor:
        item_vectors = self.item_head(candidate_features)
        scale = self.temperature.clamp_min(1e-3) * user_vectors.shape[-1] ** 0.5
        return torch.einsum("bd,bcd->bc", user_vectors, item_vectors) / scale

    def forward(
        self, history_features: Tensor, history_mask: Tensor, candidate_features: Tensor
    ) -> Tensor:
        return self.score(self.encode_user(history_features, history_mask), candidate_features)


class SIDTransformer(nn.Module):
    """Four-layer decoder-only model for autoregressive SID prediction."""

    def __init__(
        self,
        codebook_size: int = 256,
        levels: int = 4,
        hidden_dim: int = 256,
        layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        max_history: int = 50,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.levels = levels
        self.bos_id = levels * codebook_size
        self.pad_id = self.bos_id + 1
        self.vocab_size = self.pad_id + 1
        self.token = nn.Embedding(self.vocab_size, hidden_dim, padding_idx=self.pad_id)
        self.position = PositionalEncoding(hidden_dim, max_history * levels + levels + 1)
        layer = nn.TransformerEncoderLayer(
            hidden_dim, heads, hidden_dim * 4, dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, self.vocab_size, bias=False)
        self.output.weight = self.token.weight

    def offset_codes(self, codes: Tensor) -> Tensor:
        offsets = torch.arange(self.levels, device=codes.device) * self.codebook_size
        return codes + offsets

    def history_tokens(self, history_ids: Tensor, item_codes: Tensor) -> tuple[Tensor, Tensor]:
        valid = history_ids >= 0
        safe_ids = history_ids.clamp_min(0)
        codes = self.offset_codes(item_codes[safe_ids]).reshape(history_ids.shape[0], -1)
        token_mask = valid.unsqueeze(-1).expand(-1, -1, self.levels).reshape(valid.shape[0], -1)
        codes = codes.masked_fill(~token_mask, self.pad_id)
        return codes, token_mask

    def _run(self, tokens: Tensor, valid_mask: Tensor) -> Tensor:
        length = tokens.shape[1]
        causal = torch.triu(torch.ones(length, length, device=tokens.device, dtype=torch.bool), diagonal=1)
        x = self.position(self.token(tokens))
        x = self.transformer(x, mask=causal, src_key_padding_mask=~valid_mask)
        return self.output(self.norm(x))

    def training_loss(self, history_ids: Tensor, target_ids: Tensor, item_codes: Tensor) -> Tensor:
        history, history_mask = self.history_tokens(history_ids, item_codes)
        target = self.offset_codes(item_codes[target_ids])
        bos = torch.full((history.shape[0], 1), self.bos_id, device=history.device, dtype=torch.long)
        inputs = torch.cat([history, bos, target[:, :-1]], dim=1)
        valid = torch.cat([
            history_mask,
            torch.ones(history.shape[0], self.levels, device=history.device, dtype=torch.bool),
        ], dim=1)
        logits = self._run(inputs, valid)[:, -self.levels :]
        losses = []
        for level in range(self.levels):
            start = level * self.codebook_size
            losses.append(nn.functional.cross_entropy(
                logits[:, level, start : start + self.codebook_size], target[:, level] - start
            ))
        return torch.stack(losses).mean()

    @torch.no_grad()
    def score_candidates(
        self, history_ids: Tensor, candidate_ids: Tensor, item_codes: Tensor, candidate_chunk: int = 64
    ) -> Tensor:
        """Equation (10): sum autoregressive log probabilities for every SID."""
        blocks = []
        for start in range(0, candidate_ids.shape[1], candidate_chunk):
            blocks.append(self._score_candidate_block(
                history_ids, candidate_ids[:, start : start + candidate_chunk], item_codes
            ))
        return torch.cat(blocks, dim=1)

    def _score_candidate_block(self, history_ids: Tensor, candidate_ids: Tensor, item_codes: Tensor) -> Tensor:
        batch, candidates = candidate_ids.shape
        history, history_mask = self.history_tokens(history_ids, item_codes)
        history = history[:, None].expand(-1, candidates, -1).reshape(batch * candidates, -1)
        history_mask = history_mask[:, None].expand(-1, candidates, -1).reshape(batch * candidates, -1)
        candidate_codes = self.offset_codes(item_codes[candidate_ids]).reshape(batch * candidates, self.levels)
        total = torch.zeros(batch * candidates, device=history.device)
        prefix = torch.full((batch * candidates, 1), self.bos_id, device=history.device, dtype=torch.long)
        for level in range(self.levels):
            inputs = torch.cat([history, prefix], dim=1)
            valid = torch.cat([history_mask, torch.ones_like(prefix, dtype=torch.bool)], dim=1)
            start = level * self.codebook_size
            logits = self._run(inputs, valid)[:, -1, start : start + self.codebook_size].log_softmax(-1)
            token = candidate_codes[:, level]
            total += logits.gather(1, (token - start)[:, None]).squeeze(1)
            prefix = torch.cat([prefix, token[:, None]], dim=1)
        return total.reshape(batch, candidates)


class FeatureProbe(nn.Module):
    """Map four discrete codes back to the continuous feature space."""

    def __init__(self, levels: int, codebook_size: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.levels = levels
        self.embeddings = nn.ModuleList([nn.Embedding(codebook_size, hidden_dim) for _ in range(levels)])
        self.network = nn.Sequential(
            nn.Linear(levels * hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, codes: Tensor) -> Tensor:
        return self.network(torch.cat([layer(codes[:, i]) for i, layer in enumerate(self.embeddings)], dim=-1))
