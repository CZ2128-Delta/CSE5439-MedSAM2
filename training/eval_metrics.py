"""Standalone evaluation script that computes Dice and IoU on the validation set.

Unlike eval.py (which reuses the full Trainer and only reports losses), this
script directly loads the model checkpoint, runs forward passes on the
validation data, and reports per-sample and aggregate Dice / IoU metrics.

Usage (single GPU):
    python training/eval_metrics.py \
        -c configs/sam2.1_hiera_tiny512_FLARE_RECIST.yaml \
        --checkpoint /path/to/checkpoint.pt \
        --num-gpus 1
"""

import json
import logging
import os
import random
import sys
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from training.utils.checkpoint_utils import load_state_dict_into_model
from training.utils.data_utils import BatchedVideoDatapoint, collate_fn
from training.utils.train_utils import (
    get_amp_type,
    get_machine_local_and_dist_rank,
    makedir,
    register_omegaconf_resolvers,
    setup_distributed_backend,
)

os.environ["HYDRA_FULL_ERROR"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s",
    stream=sys.stdout,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_dice_iou(pred_masks: torch.Tensor, gt_masks: torch.Tensor):
    """Compute per-object Dice coefficient and IoU.

    Args:
        pred_masks: [N, H, W] binary predicted masks (bool or 0/1).
        gt_masks:   [N, H, W] binary ground-truth masks (bool or 0/1).

    Returns:
        dice: [N] tensor of Dice coefficients.
        iou:  [N] tensor of IoU values.
    """
    pred = pred_masks.flatten(1).float()
    gt = gt_masks.flatten(1).float()

    intersection = (pred * gt).sum(dim=1)
    pred_sum = pred.sum(dim=1)
    gt_sum = gt.sum(dim=1)
    union = pred_sum + gt_sum - intersection

    dice = (2.0 * intersection + 1e-8) / (pred_sum + gt_sum + 1e-8)
    iou = (intersection + 1e-8) / (union + 1e-8)
    return dice, iou


# ---------------------------------------------------------------------------
# Build validation dataloader (mirrors inject_val_config but standalone)
# ---------------------------------------------------------------------------

def build_val_dataloader(cfg, val_npz_folder, rank, world_size):
    from training.dataset.sam2_datasets import TorchTrainMixedDataset
    from training.dataset.transforms import ComposeAPI, NormalizeAPI, RandomResizeAPI, ToTensorAPI
    from training.dataset.utils import ConcatDataset, RepeatFactorWrapper
    from training.dataset.vos_dataset import VOSDataset
    from training.dataset.vos_raw_dataset import NPZRawDataset
    from training.dataset.vos_sampler import RandomUniformSampler
    from functools import partial as _partial

    transforms = [
        ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=cfg.scratch.resolution,
                    square=True,
                    consistent_transform=True,
                ),
                ToTensorAPI(),
                NormalizeAPI(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    ]

    raw_ds = NPZRawDataset(folder=val_npz_folder)
    sampler_obj = RandomUniformSampler(
        num_frames=cfg.scratch.num_frames,
        max_num_objects=cfg.scratch.max_num_objects,
    )
    vos_ds = VOSDataset(
        transforms=transforms,
        training=False,
        video_dataset=raw_ds,
        sampler=sampler_obj,
        multiplier=1,
    )
    concat_ds = ConcatDataset(datasets=[vos_ds])
    repeat_ds = RepeatFactorWrapper(dataset=concat_ds)

    repeat_ds.set_epoch(0)
    dist_sampler = DistributedSampler(
        repeat_ds, num_replicas=world_size, rank=rank, shuffle=False
    )
    loader = DataLoader(
        repeat_ds,
        batch_size=cfg.scratch.train_video_batch_size,
        sampler=dist_sampler,
        num_workers=min(cfg.scratch.num_train_workers, 8),
        pin_memory=True,
        drop_last=False,
        collate_fn=_partial(collate_fn, dict_key="all"),
    )
    return loader


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(local_rank, main_port, cfg, world_size, node_rank, master_addr,
             ckpt_path, val_npz_folder, output_dir, amp_enabled, amp_dtype):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(node_rank * cfg.launcher.gpus_per_node + local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    try:
        register_omegaconf_resolvers()
    except Exception:
        pass

    setup_distributed_backend("nccl", timeout_mins=30)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)

    rank = int(os.environ["RANK"])

    # ---- Build model ----
    model = instantiate(cfg.trainer.model, _convert_="all")
    model.to(device)
    model.eval()

    # ---- Load checkpoint (model weights only) ----
    logging.info("Loading checkpoint from %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    load_state_dict_into_model(model=model, state_dict=ckpt["model"])
    del ckpt

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )

    # ---- Build dataloader ----
    loader = build_val_dataloader(cfg, val_npz_folder, rank, world_size)
    logging.info("Validation samples: %d  (this rank sees ~%d batches)",
                 len(loader.dataset), len(loader))

    # ---- Eval loop ----
    all_dice = []
    all_iou = []
    num_batches = len(loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                m = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
                outputs = m(batch)

            gt_masks = batch.masks  # [T, O, H, W]
            num_frames = gt_masks.shape[0]

            for t in range(num_frames):
                frame_out = outputs[t]
                pred_logits_list = frame_out["multistep_pred_multimasks_high_res"]
                pred_logits = pred_logits_list[-1]  # last correction step, [N, M, H, W]

                if pred_logits.shape[1] > 1:
                    best_idx = pred_logits.argmax(dim=1, keepdim=True)
                    pred_logits = torch.gather(
                        pred_logits, 1,
                        best_idx.expand(-1, -1, pred_logits.shape[2], pred_logits.shape[3])
                    )
                pred_binary = (pred_logits.squeeze(1) > 0.0)  # [N, H, W]

                gt = gt_masks[t]  # [O, H, W]
                if gt.shape[-2:] != pred_binary.shape[-2:]:
                    gt = torch.nn.functional.interpolate(
                        gt.unsqueeze(1).float(),
                        size=pred_binary.shape[-2:],
                        mode="nearest",
                    ).squeeze(1).bool()

                gt = gt[: pred_binary.shape[0]]
                pred_binary = pred_binary[: gt.shape[0]]

                if gt.numel() == 0:
                    continue

                dice, iou = compute_dice_iou(pred_binary, gt)
                all_dice.append(dice.cpu())
                all_iou.append(iou.cpu())

            if batch_idx % 10 == 0:
                logging.info("[%d/%d] batches processed", batch_idx + 1, num_batches)

    # ---- Aggregate across this rank ----
    if all_dice:
        local_dice = torch.cat(all_dice)
        local_iou = torch.cat(all_iou)
    else:
        local_dice = torch.zeros(0)
        local_iou = torch.zeros(0)

    # ---- Gather across ranks ----
    if world_size > 1:
        local_count = torch.tensor([local_dice.numel()], device=device)
        all_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
        dist.all_gather(all_counts, local_count)
        max_count = max(c.item() for c in all_counts)

        def _pad(t, n):
            if len(t) < n:
                return torch.cat([t, torch.zeros(n - len(t))])
            return t

        padded_dice = _pad(local_dice, max_count).to(device)
        padded_iou = _pad(local_iou, max_count).to(device)
        gathered_dice = [torch.zeros_like(padded_dice) for _ in range(world_size)]
        gathered_iou = [torch.zeros_like(padded_iou) for _ in range(world_size)]
        dist.all_gather(gathered_dice, padded_dice)
        dist.all_gather(gathered_iou, padded_iou)

        all_dice_flat = torch.cat(
            [g[:c.item()] for g, c in zip(gathered_dice, all_counts)]
        ).cpu()
        all_iou_flat = torch.cat(
            [g[:c.item()] for g, c in zip(gathered_iou, all_counts)]
        ).cpu()
    else:
        all_dice_flat = local_dice
        all_iou_flat = local_iou

    # ---- Report ----
    if rank == 0:
        n = all_dice_flat.numel()
        mean_dice = all_dice_flat.mean().item() if n > 0 else 0.0
        mean_iou = all_iou_flat.mean().item() if n > 0 else 0.0
        std_dice = all_dice_flat.std().item() if n > 1 else 0.0
        std_iou = all_iou_flat.std().item() if n > 1 else 0.0

        results = {
            "num_samples": n,
            "mean_dice": round(mean_dice, 6),
            "std_dice": round(std_dice, 6),
            "mean_iou": round(mean_iou, 6),
            "std_iou": round(std_iou, 6),
        }

        logging.info("=" * 60)
        logging.info("EVALUATION RESULTS  (%d object-frame pairs)", n)
        logging.info("  Dice : %.4f ± %.4f", mean_dice, std_dice)
        logging.info("  IoU  : %.4f ± %.4f", mean_iou, std_iou)
        logging.info("=" * 60)

        makedir(output_dir)
        out_file = os.path.join(output_dir, "metrics.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        logging.info("Results saved to %s", out_file)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    initialize_config_module("sam2", version_base="1.2")
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, type=str,
        help="Hydra config name used during training",
    )
    parser.add_argument(
        "--checkpoint", required=True, type=str,
        help="Path to the training checkpoint (.pt)",
    )
    parser.add_argument(
        "--val-npz-folder", type=str,
        default="/fs/scratch/PAS3272/liu12122/MedImgSeg/FLARE-Task1-PancancerRECIST-to-3D/validation_npz",
    )
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--master-addr", type=str, default=None)
    parser.add_argument("--main-port", type=int, default=None)
    args = parser.parse_args()

    register_omegaconf_resolvers()
    cfg = compose(config_name=args.config)

    with open_dict(cfg):
        cfg.launcher.gpus_per_node = args.num_gpus
        cfg.launcher.num_nodes = args.num_nodes

    ckpt_path = os.path.abspath(args.checkpoint)
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    if args.output_path:
        output_dir = args.output_path
    else:
        output_dir = os.path.join(os.path.dirname(ckpt_path), "..", "eval_metrics")
    output_dir = os.path.abspath(output_dir)

    master_addr = args.master_addr or "localhost"
    main_port = args.main_port or random.randint(10000, 65000)

    amp_enabled = cfg.trainer.get("optim", {}).get("amp", {}).get("enabled", False)
    amp_dtype_str = cfg.trainer.get("optim", {}).get("amp", {}).get("amp_dtype", "float16")
    amp_dtype = get_amp_type(amp_dtype_str) if amp_enabled else None

    num_gpus = args.num_gpus
    world_size = num_gpus * args.num_nodes

    torch.multiprocessing.set_start_method("spawn")
    if num_gpus == 1:
        evaluate(
            local_rank=0, main_port=main_port, cfg=cfg,
            world_size=world_size, node_rank=0, master_addr=master_addr,
            ckpt_path=ckpt_path, val_npz_folder=args.val_npz_folder,
            output_dir=output_dir, amp_enabled=amp_enabled, amp_dtype=amp_dtype,
        )
    else:
        torch.multiprocessing.start_processes(
            evaluate,
            args=(main_port, cfg, world_size, 0, master_addr,
                  ckpt_path, args.val_npz_folder, output_dir,
                  amp_enabled, amp_dtype),
            nprocs=num_gpus,
            start_method="spawn",
        )


if __name__ == "__main__":
    main()
