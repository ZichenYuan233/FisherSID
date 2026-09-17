"""Compression metrics used in the FisherSID paper."""

import torch
from torch import Tensor


def ranking_kl(teacher_logits: Tensor, reconstructed_logits: Tensor) -> Tensor:
    p = teacher_logits.softmax(-1)
    log_p = teacher_logits.log_softmax(-1)
    log_q = reconstructed_logits.log_softmax(-1)
    return (p * (log_p - log_q)).sum(-1).mean()


def topk_agreement(teacher_logits: Tensor, reconstructed_logits: Tensor, k: int = 10) -> Tensor:
    left = teacher_logits.topk(k, dim=-1).indices
    right = reconstructed_logits.topk(k, dim=-1).indices
    overlap = (left.unsqueeze(-1) == right.unsqueeze(-2)).any(-1).sum(-1)
    return overlap.float().mean() / k


def collision_rate(codes: Tensor) -> Tensor:
    unique = torch.unique(codes, dim=0).shape[0]
    return codes.new_tensor(1.0 - unique / codes.shape[0], dtype=torch.float32)


def codebook_utilization(codes: Tensor, codebook_size: int) -> Tensor:
    used = torch.tensor(
        [torch.unique(codes[:, level]).numel() for level in range(codes.shape[1])],
        device=codes.device,
    )
    return used.float().mean() / codebook_size
