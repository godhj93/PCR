"""Partially overlapped point cloud matching demo with a Sinkhorn dustbin layer."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.data import PointRegistrationDataset
from testing import apply_transform


@dataclass
class SinkhornConfig:
    temperature: float = 0.1
    dustbin_score: float = -2.0
    iters: int = 100
    epsilon: float = 1e-9


@dataclass
class MatchingResult:
    transport: Tensor
    matching_matrix: Tensor
    source_to_dustbin: Tensor
    target_to_dustbin: Tensor


def log_sinkhorn_iterations(
    log_scores: Tensor,
    log_mu: Tensor,
    log_nu: Tensor,
    *,
    num_iters: int,
) -> Tensor:
    """Run Sinkhorn normalization in log-space for numerical stability."""
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)

    for _ in range(num_iters):
        u = log_mu - torch.logsumexp(log_scores + v.unsqueeze(0), dim=1)
        v = log_nu - torch.logsumexp(log_scores + u.unsqueeze(1), dim=0)

    return log_scores + u.unsqueeze(1) + v.unsqueeze(0)


def sinkhorn_with_dustbin(
    pairwise_dist: Tensor,
    *,
    config: SinkhornConfig,
    overlap_count: int,
) -> MatchingResult:
    """Compute a soft assignment matrix with an extra dustbin row/column."""
    if pairwise_dist.ndim != 2:
        raise ValueError("pairwise_dist must be a 2D tensor (num_source x num_target).")

    num_src, num_tgt = pairwise_dist.shape
    overlap_count = max(min(overlap_count, num_src, num_tgt), 0)
    unique_src = max(num_src - overlap_count, 0)
    unique_tgt = max(num_tgt - overlap_count, 0)

    scores = -pairwise_dist / max(config.temperature, config.epsilon)
    scores = scores - scores.max()

    dustbin_row = torch.full(
        (1, num_tgt), config.dustbin_score, dtype=scores.dtype, device=scores.device
    )
    extended = torch.cat([scores, dustbin_row], dim=0)
    dustbin_col = torch.full(
        (num_src + 1, 1), config.dustbin_score, dtype=scores.dtype, device=scores.device
    )
    extended = torch.cat([extended, dustbin_col], dim=1)

    mu_vals = torch.ones(num_src, dtype=scores.dtype, device=scores.device)
    nu_vals = torch.ones(num_tgt, dtype=scores.dtype, device=scores.device)
    dustbin_scale = 0.3
    if unique_src > 0:
        mu_vals[overlap_count:] = config.epsilon
    if unique_tgt > 0:
        nu_vals[overlap_count:] = config.epsilon

    mu_dustbin = torch.tensor(
        [max(float(unique_tgt) * dustbin_scale, config.epsilon)],
        dtype=scores.dtype,
        device=scores.device,
    )
    nu_dustbin = torch.tensor(
        [max(float(unique_src) * dustbin_scale, config.epsilon)],
        dtype=scores.dtype,
        device=scores.device,
    )

    mu = torch.cat([mu_vals, mu_dustbin])
    nu = torch.cat([nu_vals, nu_dustbin])
    mu = mu / mu.sum()
    nu = nu / nu.sum()

    log_mu = torch.log(mu)
    log_nu = torch.log(nu)

    log_transport = log_sinkhorn_iterations(
        extended,
        log_mu,
        log_nu,
        num_iters=config.iters,
    )
    transport = torch.exp(log_transport)

    matching_matrix = transport[:-1, :-1]
    source_to_dustbin = transport[:-1, -1]
    target_to_dustbin = transport[-1, :-1]
    return MatchingResult(
        transport=transport,
        matching_matrix=matching_matrix,
        source_to_dustbin=source_to_dustbin,
        target_to_dustbin=target_to_dustbin,
    )


def evaluate_matching(
    result: MatchingResult,
    *,
    overlap_count: int,
) -> Tuple[dict, dict]:
    """Return per-source and per-target matching statistics."""
    matching_matrix = result.matching_matrix
    num_src, num_tgt = matching_matrix.shape

    src_match_scores, src_match_idx = matching_matrix.max(dim=1)
    src_dustbin_scores = result.source_to_dustbin
    src_is_matched = src_match_scores > src_dustbin_scores

    tgt_match_scores, tgt_match_idx = matching_matrix.max(dim=0)
    tgt_dustbin_scores = result.target_to_dustbin
    tgt_is_matched = tgt_match_scores > tgt_dustbin_scores

    overlap_indices = torch.arange(overlap_count)

    src_correct_overlap = (
        (src_match_idx[:overlap_count] == overlap_indices) & src_is_matched[:overlap_count]
    )
    tgt_correct_overlap = (
        (tgt_match_idx[:overlap_count] == overlap_indices) & tgt_is_matched[:overlap_count]
    )

    unique_src_total = max(num_src - overlap_count, 0)
    unique_tgt_total = max(num_tgt - overlap_count, 0)

    stats_source = {
        "num_source": num_src,
        "pred_matched": int(src_is_matched.sum().item()),
        "overlap_accuracy": float(
            src_correct_overlap.sum().item() / max(overlap_count, 1)
        ),
        "unique_total": unique_src_total,
        "unique_to_dustbin": int((~src_is_matched)[overlap_count:].sum().item()),
        "overlap_dustbin_prob_mean": float(result.source_to_dustbin[:overlap_count].mean().item()),
        "unique_dustbin_prob_mean": float(
            result.source_to_dustbin[overlap_count:].mean().item()
        ) if unique_src_total > 0 else 0.0,
    }
    stats_target = {
        "num_target": num_tgt,
        "pred_matched": int(tgt_is_matched.sum().item()),
        "overlap_recall": float(
            tgt_correct_overlap.sum().item() / max(overlap_count, 1)
        ),
        "unique_total": unique_tgt_total,
        "unique_to_dustbin": int((~tgt_is_matched)[overlap_count:].sum().item()),
        "overlap_dustbin_prob_mean": float(result.target_to_dustbin[:overlap_count].mean().item()),
        "unique_dustbin_prob_mean": float(
            result.target_to_dustbin[overlap_count:].mean().item()
        ) if unique_tgt_total > 0 else 0.0,
    }
    return stats_source, stats_target


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    dataset = PointRegistrationDataset(
        num_samples=1,
        num_points=256,
        point_scale=1.0,
        translation_scale=0.6,
        target_noise_std=0.01,
        overlap_ratio=0.6,
        target_point_ratio=1.3,
        shape="gaussian_mixture",
        dtype=torch.float32,
        seed=42,
    )
    sample = dataset[0]

    source = sample.source.to(device).clone()
    target = sample.target.to(device).clone()
    transform = sample.transform.to(device=device, dtype=source.dtype)

    source_aligned = apply_transform(source, transform)
    separation_shift = torch.tensor([4.0, 4.0, 4.0], dtype=source.dtype, device=device)
    if sample.overlap_count < source_aligned.shape[0]:
        source_aligned[sample.overlap_count:] += separation_shift
    if sample.overlap_count < target.shape[0]:
        target[sample.overlap_count:] -= separation_shift

    pairwise_dist = torch.cdist(source_aligned, target, p=2)
    config = SinkhornConfig(temperature=0.02, dustbin_score=1.0, iters=200)
    result = sinkhorn_with_dustbin(
        pairwise_dist,
        config=config,
        overlap_count=sample.overlap_count,
    )

    stats_source, stats_target = evaluate_matching(
        result,
        overlap_count=sample.overlap_count,
    )

    print("=== Sinkhorn Dustbin Matching ===")
    print(f"Source points          : {stats_source['num_source']}")
    print(f"Target points          : {stats_target['num_target']}")
    print(f"Ground-truth overlap   : {sample.overlap_count}")
    print(f"Matched source (non-dustbin): {stats_source['pred_matched']}")
    print(f"Matched target (non-dustbin): {stats_target['pred_matched']}")
    print(f"Overlap accuracy (source view): {stats_source['overlap_accuracy']:.3f}")
    print(f"Overlap recall   (target view): {stats_target['overlap_recall']:.3f}")

    print(
        f"  mean dustbin prob (overlap sources): {stats_source['overlap_dustbin_prob_mean']:.6f}"
    )
    if stats_source["unique_total"] > 0:
        print(
            "Source uniques routed to dustbin: "
            f"{stats_source['unique_to_dustbin']} / {stats_source['unique_total']}"
        )
        print(
            f"  mean dustbin prob (unique sources): {stats_source['unique_dustbin_prob_mean']:.6f}"
        )
    else:
        print("Source uniques routed to dustbin: none (full overlap).")

    print(
        f"  mean dustbin prob (overlap targets): {stats_target['overlap_dustbin_prob_mean']:.6f}"
    )
    if stats_target["unique_total"] > 0:
        print(
            "Target uniques routed to dustbin: "
            f"{stats_target['unique_to_dustbin']} / {stats_target['unique_total']}"
        )
        print(
            f"  mean dustbin prob (unique targets): {stats_target['unique_dustbin_prob_mean']:.6f}"
        )
    else:
        print("Target uniques routed to dustbin: none (full overlap).")

    print("\nFirst 10 source match probabilities (idx, prob, dustbin_prob):")
    for idx in range(min(10, source.shape[0])):
        best_idx = torch.argmax(result.matching_matrix[idx]).item()
        best_prob = result.matching_matrix[idx, best_idx].item()
        dustbin_prob = result.source_to_dustbin[idx].item()
        print(
            f"  src[{idx:02d}] -> tgt[{best_idx:02d}]  prob={best_prob:.3e}  dustbin={dustbin_prob:.3e}"
        )


if __name__ == "__main__":
    main()
