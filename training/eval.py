# Evaluation entry-point.
# Reuses the same Trainer infrastructure as train.py but forces mode=val and
# injects a validation dataset built from the same VOSDataset/NPZRawDataset
# pipeline used for training.

import logging
import os
import random
import sys
import traceback
from argparse import ArgumentParser

import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf, open_dict

from training.utils.train_utils import makedir, register_omegaconf_resolvers

os.environ["HYDRA_FULL_ERROR"] = "1"


def single_proc_run(local_rank, main_port, cfg, world_size, node_rank, master_addr):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(node_rank * cfg.launcher.gpus_per_node + local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        register_omegaconf_resolvers()
    except Exception as e:
        logging.info(e)

    trainer = instantiate(cfg.trainer, _recursive_=False)
    trainer.run()


def single_node_runner(cfg, main_port, node_rank=0, master_addr="localhost"):
    num_proc = cfg.launcher.gpus_per_node
    world_size = cfg.launcher.gpus_per_node * cfg.launcher.num_nodes
    torch.multiprocessing.set_start_method("spawn")
    if num_proc == 1:
        single_proc_run(
            local_rank=0,
            main_port=main_port,
            cfg=cfg,
            world_size=world_size,
            node_rank=node_rank,
            master_addr=master_addr,
        )
    else:
        mp_runner = torch.multiprocessing.start_processes
        args = (main_port, cfg, world_size, node_rank, master_addr)
        mp_runner(single_proc_run, args=args, nprocs=num_proc, start_method="spawn")


def inject_val_config(cfg, val_npz_folder: str):
    """Insert ``data.val`` into the Hydra config so the Trainer can run validation.

    The validation dataset mirrors the training dataset structure but points to
    ``val_npz_folder`` and disables data augmentations that only apply during
    training (we keep resize + normalize).
    """
    with open_dict(cfg):
        cfg.trainer.mode = "val"

        cfg.trainer.data.val = {
            "_target_": "training.dataset.sam2_datasets.TorchTrainMixedDataset",
            "phases_per_epoch": 1,
            "batch_sizes": [cfg.scratch.train_video_batch_size],
            "datasets": [
                {
                    "_target_": "training.dataset.utils.RepeatFactorWrapper",
                    "dataset": {
                        "_target_": "training.dataset.utils.ConcatDataset",
                        "datasets": [
                            {
                                "_target_": "training.dataset.vos_dataset.VOSDataset",
                                "transforms": [
                                    {
                                        "_target_": "training.dataset.transforms.ComposeAPI",
                                        "transforms": [
                                            {
                                                "_target_": "training.dataset.transforms.RandomResizeAPI",
                                                "sizes": cfg.scratch.resolution,
                                                "square": True,
                                                "consistent_transform": True,
                                            },
                                            {
                                                "_target_": "training.dataset.transforms.ToTensorAPI",
                                            },
                                            {
                                                "_target_": "training.dataset.transforms.NormalizeAPI",
                                                "mean": [0.485, 0.456, 0.406],
                                                "std": [0.229, 0.224, 0.225],
                                            },
                                        ],
                                    },
                                ],
                                "training": False,
                                "video_dataset": {
                                    "_target_": "training.dataset.vos_raw_dataset.NPZRawDataset",
                                    "folder": val_npz_folder,
                                },
                                "sampler": {
                                    "_target_": "training.dataset.vos_sampler.RandomUniformSampler",
                                    "num_frames": cfg.scratch.num_frames,
                                    "max_num_objects": cfg.scratch.max_num_objects,
                                },
                                "multiplier": 1,
                            }
                        ],
                    },
                }
            ],
            "shuffle": False,
            "num_workers": cfg.scratch.num_train_workers,
            "pin_memory": True,
            "drop_last": False,
            "collate_fn": {
                "_target_": "training.utils.data_utils.collate_fn",
                "_partial_": True,
                "dict_key": "all",
            },
        }


def main(args, cfg):
    if cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = os.path.join(
            os.getcwd(), "sam2_logs", args.config
        )
    print("###################### Eval App Config ####################")
    print(OmegaConf.to_yaml(cfg))
    print("############################################################")

    makedir(cfg.launcher.experiment_log_dir)

    submitit_conf = cfg.get("submitit", None)
    assert submitit_conf is not None, "Missing submitit config"

    cfg.launcher.gpus_per_node = (
        args.num_gpus if args.num_gpus is not None else cfg.launcher.gpus_per_node
    )
    cfg.launcher.num_nodes = (
        args.num_nodes if args.num_nodes is not None else cfg.launcher.num_nodes
    )

    master_addr = args.master_addr if args.master_addr else "localhost"
    main_port = args.main_port if args.main_port else random.randint(
        submitit_conf.port_range[0], submitit_conf.port_range[1]
    )
    if "SLURM_PROCID" in os.environ:
        node_rank = int(os.environ["SLURM_PROCID"])
    else:
        node_rank = 0
    single_node_runner(cfg, main_port, node_rank=node_rank, master_addr=master_addr)


if __name__ == "__main__":
    initialize_config_module("sam2", version_base="1.2")
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, type=str,
        help="Hydra config name used during training (e.g. configs/sam2.1_hiera_tiny512_FLARE_RECIST.yaml)",
    )
    parser.add_argument(
        "--checkpoint", required=True, type=str,
        help="Path to the training checkpoint (.pt) to evaluate",
    )
    parser.add_argument(
        "--val-npz-folder", type=str,
        default="/fs/scratch/PAS3272/liu12122/MedImgSeg/FLARE-Task1-PancancerRECIST-to-3D/validation_npz",
        help="Path to the validation NPZ folder",
    )
    parser.add_argument(
        "--output-path", type=str, default=None,
        help="Directory for eval logs / outputs (defaults to a subfolder next to the checkpoint)",
    )
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--num-nodes", type=int, default=None)
    parser.add_argument("--master-addr", type=str, default=None)
    parser.add_argument("--main-port", type=int, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    args = parser.parse_args()

    register_omegaconf_resolvers()
    cfg = compose(config_name=args.config)

    inject_val_config(cfg, args.val_npz_folder)

    ckpt_path = os.path.abspath(args.checkpoint)
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    if args.output_path is not None:
        eval_output = args.output_path
    else:
        eval_output = os.path.join(os.path.dirname(ckpt_path), "..", "eval")
    eval_output = os.path.abspath(eval_output)

    with open_dict(cfg):
        cfg.launcher.experiment_log_dir = eval_output
        cfg.trainer.checkpoint.save_dir = os.path.join(eval_output, "checkpoints")
        cfg.trainer.checkpoint.resume_from = ckpt_path
        cfg.trainer.logging.log_dir = os.path.join(eval_output, "logs")
        cfg.trainer.logging.tensorboard_writer.log_dir = os.path.join(
            eval_output, "tensorboard"
        )

    if args.wandb_project or args.wandb_name or args.wandb_entity:
        with open_dict(cfg.trainer.logging):
            if args.wandb_project:
                cfg.trainer.logging.wandb_project = args.wandb_project
            if args.wandb_name:
                cfg.trainer.logging.wandb_name = args.wandb_name
            if args.wandb_entity:
                cfg.trainer.logging.wandb_entity = args.wandb_entity

    main(args, cfg)
