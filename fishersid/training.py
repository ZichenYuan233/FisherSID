"""Training, checkpointing, Fisher sampling, probes, and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from .data import make_eval_candidates, sample_negatives
from .fisher import estimate_ranking_fisher
from .metrics import ranking_kl, topk_agreement
from .models import FeatureProbe, RankingTeacher, SIDTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RunManager:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log_path = self.directory / "metrics.jsonl"

    def log(self, **values) -> None:
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(values, ensure_ascii=False) + "\n")

    def save_checkpoint(self, name: str, model: nn.Module, optimizer, epoch: int, **extra) -> None:
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, **extra},
            self.directory / name,
        )

    def resume(self, name: str, model: nn.Module, optimizer=None, device: str = "cpu") -> int:
        path = self.directory / name
        if not path.exists():
            return 0
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        if optimizer is not None and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        return int(state["epoch"]) + 1


def _move(batch: dict[str, Tensor], device: str) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_teacher(
    model: RankingTeacher,
    loader: DataLoader,
    item_features: Tensor,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    negatives: int,
    device: str,
    run: RunManager,
    seed: int,
) -> RankingTeacher:
    model.to(device)
    features = item_features.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    start = run.resume("teacher_last.pt", model, optimizer, device)
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(start, epochs):
        model.train()
        total = 0.0
        count = 0
        for raw in loader:
            batch = _move(raw, device)
            negatives_ids = sample_negatives(batch["target"].cpu(), len(features), negatives, generator).to(device)
            candidates = torch.cat([batch["target"][:, None], negatives_ids], dim=1)
            logits = model(features[batch["history"].clamp_min(0)], batch["history_mask"], features[candidates])
            loss = nn.functional.cross_entropy(logits, torch.zeros(len(candidates), dtype=torch.long, device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(candidates)
            count += len(candidates)
        run.log(stage="teacher", epoch=epoch, loss=total / max(count, 1))
        run.save_checkpoint("teacher_last.pt", model, optimizer, epoch)
    return model


def estimate_fisher_from_loader(
    teacher: RankingTeacher,
    loader: DataLoader,
    item_features: Tensor,
    *,
    sets: int = 4096,
    negatives: int = 1000,
    device: str = "cuda",
    seed: int = 0,
) -> Tensor:
    teacher.eval().to(device)
    features = item_features.to(device)
    generator = torch.Generator().manual_seed(seed)
    def fisher_batches():
        seen = 0
        for raw in loader:
            if seen >= sets:
                break
            batch = _move(raw, device)
            take = min(len(batch["target"]), sets - seen)
            history = batch["history"][:take]
            mask = batch["history_mask"][:take]
            target = batch["target"][:take]
            with torch.no_grad():
                users = teacher.encode_user(features[history.clamp_min(0)], mask)
            negative_ids = sample_negatives(target.cpu(), len(features), negatives, generator).to(device)
            candidate_ids = torch.cat([target[:, None], negative_ids], dim=1)
            yield users, features[candidate_ids].detach()
            seen += take

    def score(user_vectors: Tensor, candidate_features: Tensor) -> Tensor:
        return teacher.score(user_vectors, candidate_features)

    return estimate_ranking_fisher(score, fisher_batches()).cpu()


def train_generator(
    model: SIDTransformer,
    loader: DataLoader,
    item_codes: Tensor,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
    run: RunManager,
) -> SIDTransformer:
    model.to(device)
    codes = item_codes.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    start = run.resume("generator_last.pt", model, optimizer, device)
    for epoch in range(start, epochs):
        model.train()
        total = 0.0
        count = 0
        for raw in loader:
            batch = _move(raw, device)
            loss = model.training_loss(batch["history"], batch["target"], codes)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch["target"])
            count += len(batch["target"])
        run.log(stage="generator", epoch=epoch, loss=total / max(count, 1))
        run.save_checkpoint("generator_last.pt", model, optimizer, epoch)
    return model


def train_probe(
    probe: FeatureProbe,
    codes: Tensor,
    features: Tensor,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
    run: RunManager,
) -> FeatureProbe:
    probe.to(device)
    loader = DataLoader(TensorDataset(codes, features), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
    start = run.resume("probe_last.pt", probe, optimizer, device)
    for epoch in range(start, epochs):
        probe.train()
        total = 0.0
        for batch_codes, batch_features in loader:
            prediction = probe(batch_codes.to(device))
            loss = nn.functional.mse_loss(prediction, batch_features.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch_codes)
        run.log(stage="probe", epoch=epoch, loss=total / len(loader.dataset))
        run.save_checkpoint("probe_last.pt", probe, optimizer, epoch)
    return probe


@torch.no_grad()
def evaluate_recommender(
    generator: SIDTransformer,
    teacher: RankingTeacher,
    probe: FeatureProbe,
    loader: DataLoader,
    item_codes: Tensor,
    item_features: Tensor,
    *,
    negatives: int = 1000,
    device: str = "cuda",
    seed: int = 0,
) -> dict[str, float]:
    generator.eval().to(device)
    teacher.eval().to(device)
    probe.eval().to(device)
    codes = item_codes.to(device)
    features = item_features.to(device)
    reconstructed = probe(codes)
    totals = {"ranking_kl": 0.0, "agree_at_10": 0.0, "ndcg_at_10": 0.0, "recall_at_20": 0.0}
    count = 0
    for raw in loader:
        batch = _move(raw, device)
        candidates = make_eval_candidates(batch["target"].cpu(), len(codes), negatives, seed + count).to(device)
        scores = generator.score_candidates(batch["history"], candidates, codes)
        tie_hash = ((candidates.long() * 1103515245 + 12345) & 0x7FFFFFFF).float() / 0x7FFFFFFF
        scores = scores + tie_hash * 1e-6
        rank = scores.argsort(dim=1, descending=True)
        positive_rank = (rank == 0).nonzero(as_tuple=False)[:, 1] + 1
        totals["ndcg_at_10"] += torch.where(
            positive_rank <= 10, 1.0 / torch.log2(positive_rank.float() + 1), 0.0
        ).sum().item()
        totals["recall_at_20"] += (positive_rank <= 20).float().sum().item()
        user = teacher.encode_user(features[batch["history"].clamp_min(0)], batch["history_mask"])
        teacher_logits = teacher.score(user, features[candidates])
        reconstructed_logits = teacher.score(user, reconstructed[candidates])
        size = len(candidates)
        totals["ranking_kl"] += ranking_kl(teacher_logits, reconstructed_logits).item() * size
        totals["agree_at_10"] += topk_agreement(
            teacher_logits, reconstructed_logits, min(10, teacher_logits.shape[1])
        ).item() * size
        count += size
    result = {key: value / max(count, 1) for key, value in totals.items()}
    result["collision_rate"] = 1.0 - torch.unique(codes, dim=0).shape[0] / len(codes)
    result["codebook_utilization"] = sum(
        torch.unique(codes[:, level]).numel() / generator.codebook_size
        for level in range(generator.levels)
    ) / generator.levels
    return result
