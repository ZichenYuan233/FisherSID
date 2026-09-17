"""Qwen2-VL multimodal item feature extraction."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve
import hashlib

import torch
from torch import Tensor


class Qwen2VLItemEncoder:
    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "bfloat16"):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        torch_dtype = getattr(torch, dtype)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.process_vision_info = process_vision_info
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device
        ).eval()
        self.device = next(self.model.parameters()).device

    @torch.inference_mode()
    def encode(self, texts: list[str], images: list[object | None]) -> Tensor:
        conversations = []
        for text, image in zip(texts, images):
            content = []
            if image is not None:
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": text})
            conversations.append([{"role": "user", "content": content}])
        prompts = [self.processor.apply_chat_template(x, tokenize=False, add_generation_prompt=False) for x in conversations]
        image_inputs, video_inputs = self.process_vision_info(conversations)
        inputs = self.processor(
            text=prompts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1]
        mask = inputs["attention_mask"].unsqueeze(-1)
        return ((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).float().cpu()


def download_image(url: str, cache_dir: str | Path) -> Path | None:
    if not url:
        return None
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    suffix = Path(url.split("?")[0]).suffix or ".jpg"
    path = cache / (hashlib.sha1(url.encode()).hexdigest() + suffix)
    if not path.exists():
        urlretrieve(url, path)
    return path
