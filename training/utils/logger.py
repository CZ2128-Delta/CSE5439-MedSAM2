# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Code borrowed from TLC - https://www.internalfb.com/code/fbsource/fbcode/pytorch/tlc/torchtlc/loggers/tensorboard.py
import atexit
import functools
import logging
import math
import os
import sys
import uuid
from typing import Any, Dict, Optional, Union

from hydra.utils import instantiate

from iopath.common.file_io import g_pathmgr
from numpy import ndarray
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from training.utils.train_utils import get_machine_local_and_dist_rank, makedir

Scalar = Union[Tensor, ndarray, int, float]


def make_tensorboard_logger(log_dir: str, **writer_kwargs: Any):
    makedir(log_dir)
    summary_writer_method = SummaryWriter
    return TensorBoardLogger(
        path=log_dir, summary_writer_method=summary_writer_method, **writer_kwargs
    )


class TensorBoardWriterWrapper:
    """
    A wrapper around a SummaryWriter object.
    """

    def __init__(
        self,
        path: str,
        *args: Any,
        filename_suffix: str = None,
        summary_writer_method: Any = SummaryWriter,
        **kwargs: Any,
    ) -> None:
        """Create a new TensorBoard logger.
        On construction, the logger creates a new events file that logs
        will be written to.  If the environment variable `RANK` is defined,
        logger will only log if RANK = 0.

        NOTE: If using the logger with distributed training:
        - This logger can call collective operations
        - Logs will be written on rank 0 only
        - Logger must be constructed synchronously *after* initializing distributed process group.

        Args:
            path (str): path to write logs to
            *args, **kwargs: Extra arguments to pass to SummaryWriter
        """
        self._writer: Optional[SummaryWriter] = None
        _, self._rank = get_machine_local_and_dist_rank()
        self._path: str = path
        if self._rank == 0:
            logging.info(
                f"TensorBoard SummaryWriter instantiated. Files will be stored in: {path}"
            )
            self._writer = summary_writer_method(
                log_dir=path,
                *args,
                filename_suffix=filename_suffix or str(uuid.uuid4()),
                **kwargs,
            )
        else:
            logging.debug(
                f"Not logging meters on this host because env RANK: {self._rank} != 0"
            )
        atexit.register(self.close)

    @property
    def writer(self) -> Optional[SummaryWriter]:
        return self._writer

    @property
    def path(self) -> str:
        return self._path

    def flush(self) -> None:
        """Writes pending logs to disk."""

        if not self._writer:
            return

        self._writer.flush()

    def close(self) -> None:
        """Close writer, flushing pending logs to disk.
        Logs cannot be written after `close` is called.
        """

        if not self._writer:
            return

        self._writer.close()
        self._writer = None


class TensorBoardLogger(TensorBoardWriterWrapper):
    """
    A simple logger for TensorBoard.
    """

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        """Add multiple scalar values to TensorBoard.

        Args:
            payload (dict): dictionary of tag name and scalar value
            step (int, Optional): step value to record
        """
        if not self._writer:
            return
        for k, v in payload.items():
            self.log(k, v, step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        """Add scalar data to TensorBoard.

        Args:
            name (string): tag name used to group scalars
            data (float/int/Tensor): scalar data to log
            step (int, optional): step value to record
        """
        if not self._writer:
            return
        self._writer.add_scalar(name, data, global_step=step, new_style=True)

    def log_hparams(
        self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
    ) -> None:
        """Add hyperparameter data to TensorBoard.

        Args:
            hparams (dict): dictionary of hyperparameter names and corresponding values
            meters (dict): dictionary of name of meter and corersponding values
        """
        if not self._writer:
            return
        self._writer.add_hparams(hparams, meters)


class WandBLogger:
    """
    Weights & Biases logger (rank 0 only). Mirrors TensorBoard scalar logging.
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        dir: Optional[str] = None,
    ) -> None:
        self._wandb = None
        self._active = False
        try:
            import wandb as wandb_module
        except ImportError:
            logging.warning(
                "wandb is not installed; install with `pip install wandb` to enable W&B logging."
            )
            return

        _, rank = get_machine_local_and_dist_rank()
        if rank != 0:
            return

        self._wandb = wandb_module
        if dir:
            makedir(dir)
        self._wandb.init(
            project=project,
            name=name,
            entity=entity,
            dir=dir,
            reinit=True,
        )
        self._active = True
        run_label = name if name is not None else getattr(self._wandb.run, "name", None)
        logging.info("W&B initialized: project=%s name=%s", project, run_label)
        atexit.register(self.finish)

    def finish(self) -> None:
        if not self._active or self._wandb is None:
            return
        try:
            self._wandb.finish()
        except Exception as e:
            logging.warning("wandb.finish() failed: %s", e)
        self._active = False

    @staticmethod
    def _to_float(data: Scalar) -> float:
        if isinstance(data, Tensor):
            return float(data.detach().cpu().item())
        if isinstance(data, ndarray):
            return float(data.item() if data.ndim == 0 else data.ravel()[0])
        return float(data)

    def log(self, name: str, data: Scalar, step: int) -> None:
        if not self._active or self._wandb is None:
            return
        try:
            v = self._to_float(data)
            if not math.isfinite(v):
                return
            self._wandb.log({name: v}, step=int(step))
        except Exception as e:
            logging.debug("wandb.log skipped for %s: %s", name, e)

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        if not self._active or self._wandb is None:
            return
        out: Dict[str, float] = {}
        for k, v in payload.items():
            try:
                fv = self._to_float(v)
                if math.isfinite(fv):
                    out[k] = fv
            except Exception:
                continue
        if out:
            try:
                self._wandb.log(out, step=int(step))
            except Exception as e:
                logging.debug("wandb.log_dict failed at step %s: %s", step, e)

    def log_hparams(
        self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
    ) -> None:
        if not self._active or self._wandb is None:
            return
        try:
            flat = {
                f"hparam/{k}": self._to_float(v) for k, v in {**hparams, **meters}.items()
            }
            self._wandb.log(flat)
        except Exception as e:
            logging.debug("wandb hparams log failed: %s", e)


class Logger:
    """
    Bridges TensorBoard and optional Weights & Biases (when ``wandb_project`` is set on ``LoggingConf``).
    """

    def __init__(self, logging_conf):
        # allow turning off TensorBoard with "should_log: false" in config
        tb_config = logging_conf.tensorboard_writer
        tb_should_log = tb_config and tb_config.pop("should_log", True)
        self.tb_logger = instantiate(tb_config) if tb_should_log else None

        wp = getattr(logging_conf, "wandb_project", None)
        self.wandb_logger: Optional[WandBLogger] = None
        if wp:
            wandb_dir = getattr(logging_conf, "wandb_dir", None) or os.path.join(
                logging_conf.log_dir, "wandb"
            )
            self.wandb_logger = WandBLogger(
                project=wp,
                name=getattr(logging_conf, "wandb_name", None),
                entity=getattr(logging_conf, "wandb_entity", None),
                dir=wandb_dir,
            )

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        if self.tb_logger:
            self.tb_logger.log_dict(payload, step)
        if self.wandb_logger:
            self.wandb_logger.log_dict(payload, step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        if self.tb_logger:
            self.tb_logger.log(name, data, step)
        if self.wandb_logger:
            self.wandb_logger.log(name, data, step)

    def log_hparams(
        self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
    ) -> None:
        if self.tb_logger:
            self.tb_logger.log_hparams(hparams, meters)
        if self.wandb_logger:
            self.wandb_logger.log_hparams(hparams, meters)


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # we tune the buffering value so that the logs are updated
    # frequently.
    log_buffer_kb = 10 * 1024  # 10KB
    io = g_pathmgr.open(filename, mode="a", buffering=log_buffer_kb)
    atexit.register(io.close)
    return io


def setup_logging(
    name,
    output_dir=None,
    rank=0,
    log_level_primary="INFO",
    log_level_secondary="ERROR",
):
    """
    Setup various logging streams: stdout and file handlers.
    For file handlers, we only setup for the master gpu.
    """
    # get the filename if we want to log to the file as well
    log_filename = None
    if output_dir:
        makedir(output_dir)
        if rank == 0:
            log_filename = f"{output_dir}/log.txt"

    logger = logging.getLogger(name)
    logger.setLevel(log_level_primary)

    # create formatter
    FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s"
    formatter = logging.Formatter(FORMAT)

    # Cleanup any existing handlers
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.root.handlers = []

    # setup the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if rank == 0:
        console_handler.setLevel(log_level_primary)
    else:
        console_handler.setLevel(log_level_secondary)

    # we log to file as well if user wants
    if log_filename and rank == 0:
        file_handler = logging.StreamHandler(_cached_log_stream(log_filename))
        file_handler.setLevel(log_level_primary)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.root = logger


def shutdown_logging():
    """
    After training is done, we ensure to shut down all the logger streams.
    """
    logging.info("Shutting down loggers...")
    handlers = logging.root.handlers
    for handler in handlers:
        handler.close()
