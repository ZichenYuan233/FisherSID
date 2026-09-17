from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fishersid.config import ExperimentConfig

parser = argparse.ArgumentParser()
parser.add_argument("path", nargs="?", default="configs/beauty.json")
parser.add_argument("--dataset", default="Beauty")
args = parser.parse_args()
slug = args.dataset.lower()
config = ExperimentConfig(output_dir=f"runs/{slug}")
config.data.dataset = args.dataset
config.data.reviews_path = f"data/raw/reviews_{args.dataset}.jsonl.gz"
config.data.metadata_path = f"data/raw/meta_{args.dataset}.jsonl.gz"
config.data.processed_dir = f"data/processed/{args.dataset}"
path = Path(args.path)
config.save(path)
print(path.resolve())
