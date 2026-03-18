#!/usr/bin/env python3

import argparse
import json
import os
from typing import Any

import torch


def load_actor_bundle(run_dir: str) -> dict[str, Any]:
    return torch.load(os.path.join(run_dir, "actor_pre_loss_rank0.pt"), map_location="cpu")


def load_trainer_summary(run_dir: str) -> dict[str, Any]:
    with open(os.path.join(run_dir, "trainer_summary.json")) as f:
        return json.load(f)


def active_mask(bundle: dict[str, Any]) -> torch.Tensor:
    mask = bundle["response_mask"].bool()
    sdpo_mask = bundle.get("self_distillation_mask")
    if sdpo_mask is not None:
        mask = mask & sdpo_mask.bool().unsqueeze(1)
    return mask


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> float:
    values = tensor[mask]
    if values.numel() == 0:
        return float("nan")
    return values.float().mean().item()


def summarize_actor_bundle(name: str, bundle: dict[str, Any]) -> dict[str, Any]:
    mask = active_mask(bundle)
    summary = {
        "name": name,
        "meta": bundle.get("meta", {}),
        "response_mask_shape": list(bundle["response_mask"].shape),
        "active_sdpo_tokens": int(mask.sum().item()),
        "student_log_prob_mean": masked_mean(bundle["student_log_probs"], mask),
        "teacher_log_prob_mean": masked_mean(bundle["teacher_log_probs"], mask),
    }
    support = bundle.get("support_topk_indices")
    if support is not None:
        summary["support_topk_shape"] = list(support.shape)
    if bundle.get("student_topk_log_probs") is not None:
        summary["student_topk_shape"] = list(bundle["student_topk_log_probs"].shape)
    if bundle.get("teacher_topk_log_probs") is not None:
        summary["teacher_topk_shape"] = list(bundle["teacher_topk_log_probs"].shape)
    return summary


def compare_actor_bundles(fsdp: dict[str, Any], megatron: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    fsdp_mask = active_mask(fsdp)
    megatron_mask = active_mask(megatron)

    result["response_mask_same_shape"] = list(fsdp["response_mask"].shape) == list(megatron["response_mask"].shape)
    if result["response_mask_same_shape"]:
        result["response_mask_equal_fraction"] = (
            (fsdp["response_mask"] == megatron["response_mask"]).float().mean().item()
        )
        joint_mask = fsdp_mask & megatron_mask
    else:
        joint_mask = None

    result["active_sdpo_tokens"] = {
        "fsdp": int(fsdp_mask.sum().item()),
        "megatron": int(megatron_mask.sum().item()),
    }

    for key in ["student_log_probs", "teacher_log_probs"]:
        if list(fsdp[key].shape) == list(megatron[key].shape):
            diff = (fsdp[key] - megatron[key]).abs()
            result[f"{key}_mean_abs_diff"] = diff.mean().item()
            if joint_mask is not None:
                result[f"{key}_masked_mean_abs_diff"] = masked_mean(diff, joint_mask)

    fsdp_support = fsdp.get("support_topk_indices")
    megatron_support = megatron.get("support_topk_indices")
    if fsdp_support is not None and megatron_support is not None and list(fsdp_support.shape) == list(megatron_support.shape):
        result["support_index_exact_match_fraction"] = (fsdp_support == megatron_support).float().mean().item()
        fsdp_set = fsdp_support.sort(dim=-1).values
        megatron_set = megatron_support.sort(dim=-1).values
        result["support_set_exact_match_fraction"] = (fsdp_set == megatron_set).all(dim=-1).float().mean().item()

    return result


def summarize_trainer(label: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        "global_step": summary.get("global_step"),
        "experiment_name": summary.get("experiment_name"),
        "actor_strategy": summary.get("actor_strategy"),
        "teacher_scoring_mode": summary.get("teacher_scoring_mode"),
        "trainer_batch": summary.get("trainer_batch"),
        "teacher_batch": summary.get("teacher_batch"),
        "teacher_targets": summary.get("teacher_targets"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare one FSDP and one Megatron SDPO debug dump.")
    parser.add_argument("--fsdp-dir", required=True, help="Step dump directory for the FSDP run")
    parser.add_argument("--megatron-dir", required=True, help="Step dump directory for the Megatron run")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    fsdp_trainer = load_trainer_summary(args.fsdp_dir)
    megatron_trainer = load_trainer_summary(args.megatron_dir)
    fsdp_actor = load_actor_bundle(args.fsdp_dir)
    megatron_actor = load_actor_bundle(args.megatron_dir)

    report = {
        "trainer": {
            "fsdp": summarize_trainer("fsdp", fsdp_trainer),
            "megatron": summarize_trainer("megatron", megatron_trainer),
        },
        "actor": {
            "fsdp": summarize_actor_bundle("fsdp", fsdp_actor),
            "megatron": summarize_actor_bundle("megatron", megatron_actor),
            "comparison": compare_actor_bundles(fsdp_actor, megatron_actor),
        },
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
