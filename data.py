"""Amazon review preprocessing, chronological splits, and candidate sampling."""

from __future__ import annotations

import gzip
import ast
import json
from pathlib import Path
import random
from typing import Iterable, Iterator

import torch
from torch.utils.data import Dataset


def _json_lines(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    yield ast.literal_eval(line)


def iterative_k_core(interactions: list[tuple[str, str, int]], k: int = 5) -> list[tuple[str, str, int]]:
    """Apply user-and-item k-core filtering until the set stops changing."""
    current = interactions
    while True:
        user_count: dict[str, int] = {}
        item_count: dict[str, int] = {}
        for user, item, _ in current:
            user_count[user] = user_count.get(user, 0) + 1
            item_count[item] = item_count.get(item, 0) + 1
        filtered = [x for x in current if user_count[x[0]] >= k and item_count[x[1]] >= k]
        if len(filtered) == len(current):
            return filtered
        current = filtered


def preprocess_amazon(
    reviews_path: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
    *,
    min_interactions: int = 5,
) -> dict:
    """Create indexed interactions and leave-two-out chronological splits."""
    interactions = []
    for row in _json_lines(reviews_path):
        user = str(row.get("reviewerID", row.get("user_id")))
        item = str(row.get("asin", row.get("parent_asin", row.get("item_id"))))
        timestamp = int(row.get("unixReviewTime", row.get("timestamp", 0)))
        if user != "None" and item != "None":
            interactions.append((user, item, timestamp))
    interactions = iterative_k_core(interactions, min_interactions)
    users = sorted({x[0] for x in interactions})
    items = sorted({x[1] for x in interactions})
    user_to_id = {value: index for index, value in enumerate(users)}
    item_to_id = {value: index for index, value in enumerate(items)}

    sequences: list[list[int]] = [[] for _ in users]
    per_user: list[list[tuple[int, int]]] = [[] for _ in users]
    for user, item, timestamp in interactions:
        per_user[user_to_id[user]].append((timestamp, item_to_id[item]))
    for user_id, history in enumerate(per_user):
        sequences[user_id] = [item for _, item in sorted(history, key=lambda x: (x[0], x[1]))]

    metadata = {}
    wanted = set(items)
    for row in _json_lines(metadata_path):
        asin = str(row.get("asin", row.get("parent_asin")))
        if asin in wanted:
            title = str(row.get("title", ""))
            description = row.get("description", "")
            if isinstance(description, list):
                description = " ".join(map(str, description))
            images = row.get("imageURLHigh", row.get("images", row.get("image", [])))
            if isinstance(images, str):
                images = [images]
            metadata[item_to_id[asin]] = {
                "asin": asin, "title": title, "description": str(description), "images": images,
            }

    split = {
        "train": [seq[:-2] for seq in sequences],
        "validation": [seq[-2] for seq in sequences],
        "test": [seq[-1] for seq in sequences],
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(split, output / "splits.pt")
    (output / "users.json").write_text(json.dumps(user_to_id), encoding="utf-8")
    (output / "items.json").write_text(json.dumps(item_to_id), encoding="utf-8")
    with (output / "metadata.jsonl").open("w", encoding="utf-8") as stream:
        for item_id in range(len(items)):
            stream.write(json.dumps({"item_id": item_id, **metadata.get(item_id, {})}, ensure_ascii=False) + "\n")
    stats = {"users": len(users), "items": len(items), "interactions": len(interactions)}
    (output / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


class NextItemDataset(Dataset):
    """All train prefixes or a single validation/test prefix per user."""

    def __init__(self, splits: dict, mode: str = "train", max_history: int = 50):
        self.examples: list[tuple[list[int], int, int]] = []
        train = splits["train"]
        if mode == "train":
            for user, sequence in enumerate(train):
                for position in range(1, len(sequence)):
                    self.examples.append((sequence[max(0, position - max_history):position], sequence[position], user))
        else:
            for user, sequence in enumerate(train):
                prefix = list(sequence)
                if mode == "test":
                    prefix.append(splits["validation"][user])
                target = splits[mode][user]
                self.examples.append((prefix[-max_history:], target, user))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[list[int], int, int]:
        return self.examples[index]


def collate_next_item(batch: list[tuple[list[int], int, int]]) -> dict[str, torch.Tensor]:
    max_length = max(len(x[0]) for x in batch)
    history = torch.full((len(batch), max_length), -1, dtype=torch.long)
    mask = torch.zeros_like(history, dtype=torch.bool)
    for row, (sequence, _, _) in enumerate(batch):
        history[row, -len(sequence):] = torch.tensor(sequence)
        mask[row, -len(sequence):] = True
    return {
        "history": history,
        "history_mask": mask,
        "target": torch.tensor([x[1] for x in batch]),
        "user": torch.tensor([x[2] for x in batch]),
    }


def sample_negatives(targets: torch.Tensor, num_items: int, count: int, generator: torch.Generator) -> torch.Tensor:
    """Uniform negatives, rejecting the positive item."""
    negatives = torch.randint(num_items - 1, (targets.shape[0], count), generator=generator)
    return negatives + (negatives >= targets[:, None]).long()


def make_eval_candidates(targets: torch.Tensor, num_items: int, negatives: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for target in targets.tolist():
        values = torch.randperm(num_items - 1, generator=generator)[:negatives]
        rows.append(values + (values >= target).long())
    return torch.cat([targets[:, None], torch.stack(rows)], dim=1)
