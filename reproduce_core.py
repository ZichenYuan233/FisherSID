"""Run a deterministic synthetic reproduction of the FisherSID core method."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

# Make ``python scripts/reproduce_core.py`` work from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fishersid import FisherSID, estimate_ranking_fisher
from fishersid.metrics import collision_rate, ranking_kl, topk_agreement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/synthetic"))
    parser.add_argument("--items", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--users", type=int, default=256)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--codebook-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    # Anisotropic catalog plus a bilinear item-separable teacher.
    scale = torch.linspace(2.0, 0.2, args.dim)
    item_features = torch.randn(args.items, args.dim) * scale
    user_context = torch.randn(args.users, args.dim)
    candidate_ids = torch.randint(args.items, (args.users, args.candidates))
    candidate_features = item_features[candidate_ids]

    def teacher(users: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return (users[:, None, :] * features).sum(-1) / args.dim**0.5

    batch_size = 32
    batches = [
        (user_context[start : start + batch_size], candidate_features[start : start + batch_size])
        for start in range(0, args.users, batch_size)
    ]
    fisher = estimate_ranking_fisher(teacher, batches)
    tokenizer = FisherSID(
        rank=args.rank, levels=4, codebook_size=args.codebook_size,
        max_iters=100, seed=args.seed,
    ).fit(item_features, fisher)
    codes = tokenizer.encode(item_features)
    reconstructed = tokenizer.reconstruct_features(codes)

    original_logits = teacher(user_context, candidate_features)
    reconstructed_logits = teacher(user_context, reconstructed[candidate_ids])
    metrics = {
        "ranking_kl": float(ranking_kl(original_logits, reconstructed_logits)),
        "agree_at_10": float(topk_agreement(original_logits, reconstructed_logits, k=10)),
        "collision_rate": float(collision_rate(codes)),
    }
    tokenizer.save(str(args.output / "fishersid.pt"))
    torch.save({"codes": codes, "fisher": fisher, "metrics": metrics}, args.output / "results.pt")
    print("Synthetic FisherSID reproduction")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")
    print(f"saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
