from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fishersid.baselines import LinearRQTokenizer, train_neural_baseline
from fishersid.config import ExperimentConfig
from fishersid.data import NextItemDataset, collate_next_item
from fishersid.models import FeatureProbe, RankingTeacher, SIDTransformer
from fishersid.pipeline import FisherSID
from fishersid.training import (
    RunManager, estimate_fisher_from_loader, evaluate_recommender, set_seed,
    train_generator, train_probe, train_teacher,
)


def neighborhood_targets(splits: dict, features: torch.Tensor) -> torch.Tensor:
    target = torch.zeros_like(features)
    count = torch.zeros(len(features), 1)
    for sequence in splits["train"]:
        for left, right in zip(sequence[:-1], sequence[1:]):
            target[left] += features[right]
            target[right] += features[left]
            count[left] += 1
            count[right] += 1
    return target / count.clamp_min(1)


def tokenize(method, features, fisher, teacher, auxiliary, config, device):
    tc = config.tokenizer
    common = dict(levels=tc.levels, codebook_size=tc.codebook_size, max_iters=tc.kmeans_iters, seed=0)
    if method == "fishersid":
        model = FisherSID(rank=tc.rank, rho=tc.rho, **common).fit(features, fisher)
        return model.encode(features), model.state_dict()
    if method == "fisher_rq":
        model = FisherSID(rank=tc.rank, rho=0.0, **common).fit(features, fisher)
        return model.encode(features), model.state_dict()
    if method in {"raw_rq", "pca_rq"}:
        model = LinearRQTokenizer(method, rank=tc.rank, **common).fit(features)
        return model.encode(features), model
    if method in {"tiger", "letter", "mme_sid", "musicrec", "pit"}:
        aux = None if method == "tiger" else auxiliary
        model, codes = train_neural_baseline(
            method, features, auxiliary=aux, latent_dim=tc.rank, levels=tc.levels,
            codebook_size=tc.codebook_size, epochs=tc.neural_epochs,
            batch_size=tc.neural_batch_size, learning_rate=tc.neural_learning_rate,
            device=device,
        )
        return codes, model.state_dict()
    return LinearRQTokenizer(method, rank=tc.rank, **common).fit(features).encode(features), None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    device = config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu"
    processed = Path(config.data.processed_dir)
    splits = torch.load(processed / "splits.pt", weights_only=True)
    features = torch.load(processed / "item_features.pt", map_location="cpu", weights_only=True).float()
    train_data = NextItemDataset(splits, "train", config.teacher.max_history)
    validation_data = NextItemDataset(splits, "validation", config.teacher.max_history)
    test_data = NextItemDataset(splits, "test", config.teacher.max_history)
    auxiliary = neighborhood_targets(splits, features)
    all_results = []

    for seed in config.seeds:
        set_seed(seed)
        seed_dir = Path(config.output_dir) / f"seed_{seed}"
        teacher_run = RunManager(seed_dir / "teacher")
        train_loader = DataLoader(
            train_data, batch_size=config.teacher.batch_size, shuffle=True,
            collate_fn=collate_next_item,
        )
        generator_loader = DataLoader(
            train_data, batch_size=config.generator.batch_size, shuffle=True,
            collate_fn=collate_next_item,
        )
        validation_loader = DataLoader(
            validation_data, batch_size=min(32, config.teacher.batch_size), shuffle=False,
            collate_fn=collate_next_item,
        )
        test_loader = DataLoader(
            test_data, batch_size=min(32, config.teacher.batch_size), shuffle=False,
            collate_fn=collate_next_item,
        )
        fisher_loader = DataLoader(
            train_data, batch_size=config.tokenizer.fisher_batch_size, shuffle=True,
            collate_fn=collate_next_item,
        )
        t = config.teacher
        teacher = RankingTeacher(
            t.feature_dim, t.hidden_dim, t.layers, t.heads, t.dropout, t.max_history
        )
        train_teacher(
            teacher, train_loader, features, epochs=t.epochs, learning_rate=t.learning_rate,
            weight_decay=t.weight_decay, negatives=t.train_negatives, device=device,
            run=teacher_run, seed=seed,
        )
        fisher_path = seed_dir / "fisher.pt"
        if fisher_path.exists():
            fisher = torch.load(fisher_path, map_location="cpu", weights_only=True)
        else:
            fisher = estimate_fisher_from_loader(
                teacher, fisher_loader, features, sets=config.tokenizer.fisher_sets,
                negatives=config.tokenizer.fisher_negatives, device=device, seed=seed,
            )
            fisher_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(fisher, fisher_path)

        for method in config.methods:
            set_seed(seed)
            method_dir = seed_dir / method
            run = RunManager(method_dir)
            codes_path = method_dir / "codes.pt"
            if codes_path.exists():
                codes = torch.load(codes_path, map_location="cpu", weights_only=True)
            else:
                codes, tokenizer_state = tokenize(method, features, fisher, teacher, auxiliary, config, device)
                method_dir.mkdir(parents=True, exist_ok=True)
                torch.save(codes, codes_path)
                torch.save(tokenizer_state, method_dir / "tokenizer.pt")

            g = config.generator
            generator = SIDTransformer(
                config.tokenizer.codebook_size, config.tokenizer.levels, g.hidden_dim,
                g.layers, g.heads, g.dropout, g.max_history,
            )
            train_generator(
                generator, generator_loader, codes, epochs=g.epochs, learning_rate=g.learning_rate,
                weight_decay=g.weight_decay, device=device, run=run,
            )
            p = config.probe
            probe = FeatureProbe(config.tokenizer.levels, config.tokenizer.codebook_size, p.hidden_dim, features.shape[1])
            train_probe(
                probe, codes, features, epochs=p.epochs, batch_size=p.batch_size,
                learning_rate=p.learning_rate, weight_decay=p.weight_decay, device=device, run=run,
            )
            validation_metrics = evaluate_recommender(
                generator, teacher, probe, validation_loader, codes, features,
                negatives=config.data.candidate_negatives, device=device, seed=seed,
            )
            test_metrics = evaluate_recommender(
                generator, teacher, probe, test_loader, codes, features,
                negatives=config.data.candidate_negatives, device=device, seed=seed + 100000,
            )
            row = {"seed": seed, "method": method, **test_metrics, "validation": validation_metrics}
            (method_dir / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            all_results.append(row)
            print(row)

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
