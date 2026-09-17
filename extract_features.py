from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fishersid.config import ExperimentConfig
from fishersid.features import Qwen2VLItemEncoder, download_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    processed = Path(config.data.processed_dir)
    rows = [json.loads(line) for line in (processed / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    encoder = Qwen2VLItemEncoder(config.features.model_name, config.device, config.features.dtype)
    output = []
    cache = processed / "images"
    for start in range(0, len(rows), config.features.batch_size):
        batch = rows[start : start + config.features.batch_size]
        texts = [(row.get("title", "") + "\n" + row.get("description", ""))[:4000] for row in batch]
        images = []
        for row in batch:
            urls = row.get("images", [])
            url = urls[0] if urls else ""
            if isinstance(url, dict):
                url = url.get("large", url.get("hi_res", url.get("thumb", "")))
            path = download_image(url, cache) if url else None
            images.append(str(path) if path else None)
        output.append(encoder.encode(texts, images))
        print(f"{min(start + len(batch), len(rows))}/{len(rows)}")
    torch.save(torch.cat(output), processed / "item_features.pt")


if __name__ == "__main__":
    main()
