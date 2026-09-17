from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fishersid.config import ExperimentConfig
from fishersid.data import preprocess_amazon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    stats = preprocess_amazon(
        config.data.reviews_path,
        config.data.metadata_path,
        config.data.processed_dir,
        min_interactions=config.data.min_interactions,
    )
    print(stats)


if __name__ == "__main__":
    main()
