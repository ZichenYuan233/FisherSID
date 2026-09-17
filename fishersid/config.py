"""Experiment configuration with JSON serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class DataConfig:
    dataset: str = "Beauty"
    reviews_path: str = "data/raw/reviews_Beauty.jsonl.gz"
    metadata_path: str = "data/raw/meta_Beauty.jsonl.gz"
    processed_dir: str = "data/processed/Beauty"
    min_interactions: int = 5
    candidate_negatives: int = 1000


@dataclass
class FeatureConfig:
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    output_dim: int = 1536
    batch_size: int = 4
    max_text_length: int = 256
    dtype: str = "bfloat16"


@dataclass
class TeacherConfig:
    feature_dim: int = 1536
    hidden_dim: int = 256
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    max_history: int = 50
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    train_negatives: int = 128


@dataclass
class TokenizerConfig:
    fisher_sets: int = 4096
    fisher_negatives: int = 1000
    fisher_batch_size: int = 4
    rank: int = 64
    rho: float = 0.5
    levels: int = 4
    codebook_size: int = 256
    kmeans_iters: int = 100
    neural_epochs: int = 100
    neural_batch_size: int = 512
    neural_learning_rate: float = 1e-3


@dataclass
class GeneratorConfig:
    hidden_dim: int = 256
    layers: int = 4
    heads: int = 4
    dropout: float = 0.1
    max_history: int = 50
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4


@dataclass
class ProbeConfig:
    hidden_dim: int = 512
    epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5


@dataclass
class ExperimentConfig:
    output_dir: str = "runs/beauty"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    methods: list[str] = field(default_factory=lambda: [
        "raw_rq", "pca_rq", "tiger", "letter", "mme_sid", "musicrec",
        "pit", "fisher_rq", "fishersid",
    ])
    device: str = "cuda"
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            output_dir=raw.get("output_dir", "runs/beauty"),
            seeds=raw.get("seeds", [0, 1, 2]),
            methods=raw.get("methods", cls().methods),
            device=raw.get("device", "cuda"),
            data=DataConfig(**raw.get("data", {})),
            features=FeatureConfig(**raw.get("features", {})),
            teacher=TeacherConfig(**raw.get("teacher", {})),
            tokenizer=TokenizerConfig(**raw.get("tokenizer", {})),
            generator=GeneratorConfig(**raw.get("generator", {})),
            probe=ProbeConfig(**raw.get("probe", {})),
        )
