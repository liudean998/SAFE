"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
import time
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt
import torch
import torch_geometric
from tqdm import tqdm

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.common.relaxation.ml_relaxation import ml_relax
from ocpmodels.common.utils import check_traj_files, get_pbc_distances
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.scaling.util import ensure_fitted
from ocpmodels.trainers.base_trainer import BaseTrainer

@registry.register_trainer("is2rs")
class Is2RsTrainer(BaseTrainer):
    """
    Trainer class for the Structure to Energy & Force (S2EF) and Initial State to
    Relaxed State (IS2RS) tasks.

    .. note::

        Examples of configurations for task, model, dataset and optimizer
        can be found in `configs/ocp_s2ef <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2re/>`_
        and `configs/ocp_is2rs <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2rs/>`_.

    Args:
        task (dict): Task configuration.
        model (dict): Model configuration.
        dataset (dict): Dataset configuration. The dataset needs to be a SinglePointLMDB dataset.
        optimizer (dict): Optimizer configuration.
        identifier (str): Experiment identifier that is appended to log directory.
        run_dir (str, optional): Path to the run directory where logs are to be saved.
            (default: :obj:`None`)
        is_debug (bool, optional): Run in debug mode.
            (default: :obj:`False`)
        is_hpo (bool, optional): Run hyperparameter optimization with Ray Tune.
            (default: :obj:`False`)
        print_every (int, optional): Frequency of printing logs.
            (default: :obj:`100`)
        seed (int, optional): Random number seed.
            (default: :obj:`None`)
        logger (str, optional): Type of logger to be used.
            (default: :obj:`tensorboard`)
        local_rank (int, optional): Local rank of the process, only applicable for distributed training.
            (default: :obj:`0`)
        amp (bool, optional): Run using automatic mixed precision.
            (default: :obj:`False`)
        slurm (dict): Slurm configuration. Currently just for keeping track.
            (default: :obj:`{}`)
    """

    def __init__(
        self,
        task,
        model,
        dataset,
        optimizer,
        identifier,
        normalizer=None,
        timestamp_id: Optional[str] = None,
        run_dir: Optional[str] = None,
        is_debug: bool = False,
        is_hpo: bool = False,
        print_every: int = 100,
        seed: Optional[int] = None,
        logger: str = "tensorboard",
        local_rank: int = 0,
        amp: bool = False,
        cpu: bool = False,
        slurm={},
        noddp: bool = False,
    ) -> None:
        super().__init__(
            task=task,
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            identifier=identifier,
            normalizer=normalizer,
            timestamp_id=timestamp_id,
            run_dir=run_dir,
            is_debug=is_debug,
            is_hpo=is_hpo,
            print_every=print_every,
            seed=seed,
            logger=logger,
            local_rank=local_rank,
            amp=amp,
            cpu=cpu,
            name="is2rs",
            slurm=slurm,
            noddp=noddp,
        )
        self.v_target=None

    def load_task(self) -> None:
        logging.info(f"Loading dataset: {self.config['task']['dataset']}")
        self.num_targets = 1

    # Takes in a new data source and generates predictions on it.
    @torch.no_grad()
    def predict(
        self,
        data_loader,
        per_image: bool = True,
        results_file=None,
        disable_tqdm: bool = False,
    ):
        ensure_fitted(self._unwrapped_model, warn=True)
        if distutils.is_master() and not disable_tqdm:
            logging.info("Predicting on test.")
        assert isinstance(
            data_loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = distutils.get_rank()

        if isinstance(data_loader, torch_geometric.data.Batch):
            data_loader = [[data_loader]]

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        predictions = {"id": [], "positions": []}
        for i, batch_list in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch_list)
            predictions['positions'].extend(out['positions'].cpu().detach())
            if per_image:
                if isinstance(batch_list[0].sid, list):
                    predictions["id"].extend(
                        [str(i) for i in batch_list[0].sid]
                    )
                else:
                    predictions["id"].extend(
                        [str(i) for i in batch_list[0].sid.tolist()]
                    )
            else:
                if isinstance(batch_list[0].sid, list):
                    predictions["id"].extend(
                        [str(i) for i in batch_list[0].sid]
                    )
                else:
                    predictions["id"].extend(
                        [str(i) for i in batch_list[0].sid.tolist()]
                    )
        self.save_results(
            predictions, results_file, keys=["positions"]
        )
        if self.ema:
            self.ema.restore()

        return predictions

    def update_best(
        self,
        primary_metric,
        val_metrics,
        disable_eval_tqdm: bool = True,
    ) -> None:
        if (
            "mae" in primary_metric
            and val_metrics[primary_metric]["metric"] < self.best_val_metric
        ) or (
            "mae" not in primary_metric
            and val_metrics[primary_metric]["metric"] > self.best_val_metric
        ):
            self.best_val_metric = val_metrics[primary_metric]["metric"]
            self.save(
                metrics=val_metrics,
                checkpoint_file="best_checkpoint.pt",
                training_state=False,
            )
            if self.test_loader is not None:
                self.predict(
                    self.test_loader,
                    results_file="predictions",
                    disable_tqdm=disable_eval_tqdm,
                    per_image=False
                )

    def train(self, disable_eval_tqdm: bool = False) -> None:

        ensure_fitted(self._unwrapped_model, warn=True)

        eval_every = self.config["optim"].get(
            "eval_every", len(self.train_loader)
        )
        checkpoint_every = self.config["optim"].get(
            "checkpoint_every", eval_every
        )
        primary_metric = self.config["task"].get(
            "primary_metric", self.evaluator.task_primary_metric['is2rs']
        )
        if (
            not hasattr(self, "primary_metric")
            or self.primary_metric != primary_metric
        ):
            self.best_val_metric = 1e9 if "mae" in primary_metric else -1.0
        else:
            primary_metric = self.primary_metric
        self.metrics = {}

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)

        for epoch_int in range(
            start_epoch, self.config["optim"]["max_epochs"]
        ):
            self.train_sampler.set_epoch(epoch_int)
            skip_steps = self.step % len(self.train_loader)
            train_loader_iter = iter(self.train_loader)

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()

                # Get a batch.
                batch = next(train_loader_iter)

                # is_back = True
                # Forward, loss, backward.
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    out = self._forward(batch)
                    loss = self._compute_loss(out, batch)

                loss = self.scaler.scale(loss) if self.scaler else loss

                self._backward(loss)
                scale = self.scaler.get_scale() if self.scaler else 1.0

                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out,
                    batch,
                    self.evaluator,
                    self.metrics,
                )
                self.metrics = self.evaluator.update(
                    "loss", loss.item() / scale, self.metrics
                )

                # Log metrics.
                log_dict = {k: self.metrics[k]["metric"] for k in self.metrics}
                log_dict.update(
                    {
                        "lr": self.scheduler.get_lr(),
                        "epoch": self.epoch,
                        "step": self.step,
                    }
                )
                if (
                    self.step % self.config["cmd"]["print_every"] == 0
                    and distutils.is_master()
                    and not self.is_hpo
                ):
                    log_str = [
                        "{}: {:.2e}".format(k, v) for k, v in log_dict.items()
                    ]
                    logging.info(", ".join(log_str))
                    self.metrics = {}

                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split="train",
                    )

                if (
                    checkpoint_every != -1
                    and self.step % checkpoint_every == 0
                ):
                    self.save(
                        checkpoint_file="checkpoint.pt", training_state=True
                    )

                # Evaluate on val set every `eval_every` iterations.
                if self.step % eval_every == 0:
                    if self.val_loader is not None:
                        val_metrics = self.validate(
                            split="val",
                            disable_tqdm=disable_eval_tqdm,
                        )
                        self.update_best(
                            primary_metric,
                            val_metrics,
                            disable_eval_tqdm=disable_eval_tqdm,
                        )
                        if self.is_hpo:
                            self.hpo_update(
                                self.epoch,
                                self.step,
                                self.metrics,
                                val_metrics,
                            )
                    if self.config["task"].get("eval_relaxations", False):
                        if "relax_dataset" not in self.config["task"]:
                            logging.warning(
                                "Cannot evaluate relaxations, relax_dataset not specified"
                            )
                        else:
                            self.run_relaxations()

                if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                    if self.step % eval_every == 0:
                        self.scheduler.step(
                            metrics=val_metrics[primary_metric]["metric"],
                        )
                else:
                    self.scheduler.step()

            torch.cuda.empty_cache()

            if checkpoint_every == -1:
                self.save(checkpoint_file="checkpoint.pt", training_state=True)

        self.train_dataset.close_db()
        if self.config.get("val_dataset", False):
            self.val_dataset.close_db()
        if self.config.get("test_dataset", False):
            self.test_dataset.close_db()

    def _forward(self, batch_list):
        # forward pass.
        # out = self.model(batch_list)
        out_p, main_graph = self.model(batch_list)
        out = {
            "positions": out_p if out_p is not None else torch.tensor([]),
        }
        if self.config["task"].get("train_on_free_atoms", True):
            batch = batch_list[0]
            # 如果使用完整结构，这里需要对边和原子进行筛选
            tags = batch.tags
            atom_mask = tags != 0
            out['positions'] = out['positions'][atom_mask]
        return out

    def _compute_loss(self, out, batch_list) -> int:
        pos_target = torch.cat([batch.pos_relaxed.to(self.device) for batch in batch_list])
        pos_origin = torch.cat([batch.pos.to(self.device) for batch in batch_list])
        p_mult = self.config["optim"].get("pos_coefficient", 1)
        if self.config["task"].get("train_on_free_atoms", True):
        # ----------- 坐标损失
            tags = batch_list[0].tags
            atom_mask = tags != 0
            p_loss = p_mult*self.loss_fn['positions'](out['positions']+pos_origin[atom_mask],
                                                   pos_target[atom_mask])
        else:
            p_loss = p_mult*self.loss_fn['vector'](out['positions']+pos_origin,
                                                   pos_target)
        # 先以相同权重生成loss
        loss = p_loss.clone()
        return loss

    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        natoms = torch.cat(
            [batch.natoms.to(self.device) for batch in batch_list], dim=0
        )
        pos_origin = torch.cat([batch.pos.to(self.device) for batch in batch_list])
        p_target = torch.cat(
            [batch.pos_relaxed.to(self.device) for batch in batch_list], dim=0
        )
        if self.config["task"].get("train_on_free_atoms", True):
            tags = batch_list[0].tags
            atom_mask = tags != 0
            new_natoms = []
            start = 0
            for natom in natoms:
                end = start + natom
                # 对每个结构的原子进行 atom_mask 筛选
                mask = atom_mask[start:end]
                new_natoms.append(mask.sum())
                start = end

            natoms = torch.stack(new_natoms)
            out['positions'] = out['positions'] + pos_origin[atom_mask]
            p_target = p_target[atom_mask]
        else:
            out['positions'] = out['positions'] + pos_origin

        target = {
            "natoms": natoms,
            'positions':p_target,
            "cell": batch_list[0].cell,
            "pbc": torch.tensor([[True, True, False]])*len(batch_list[0].cell)
        }
        out['natoms'] = natoms
        out['cell'] = batch_list[0].cell
        out['pbc'] = torch.tensor([[True, True, False]])*len(out['cell'])

        metrics = evaluator.eval(out, target, prev_metrics=metrics)
        return metrics


@registry.register_trainer("is2rsv")
class Is2RsTrainer(BaseTrainer):
    """
    Trainer class for the Structure to Energy & Force (S2EF) and Initial State to
    Relaxed State (IS2RS) tasks.

    .. note::

        Examples of configurations for task, model, dataset and optimizer
        can be found in `configs/ocp_s2ef <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2re/>`_
        and `configs/ocp_is2rs <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2rs/>`_.

    Args:
        task (dict): Task configuration.
        model (dict): Model configuration.
        dataset (dict): Dataset configuration. The dataset needs to be a SinglePointLMDB dataset.
        optimizer (dict): Optimizer configuration.
        identifier (str): Experiment identifier that is appended to log directory.
        run_dir (str, optional): Path to the run directory where logs are to be saved.
            (default: :obj:`None`)
        is_debug (bool, optional): Run in debug mode.
            (default: :obj:`False`)
        is_hpo (bool, optional): Run hyperparameter optimization with Ray Tune.
            (default: :obj:`False`)
        print_every (int, optional): Frequency of printing logs.
            (default: :obj:`100`)
        seed (int, optional): Random number seed.
            (default: :obj:`None`)
        logger (str, optional): Type of logger to be used.
            (default: :obj:`tensorboard`)
        local_rank (int, optional): Local rank of the process, only applicable for distributed training.
            (default: :obj:`0`)
        amp (bool, optional): Run using automatic mixed precision.
            (default: :obj:`False`)
        slurm (dict): Slurm configuration. Currently just for keeping track.
            (default: :obj:`{}`)
    """

    def __init__(
        self,
        task,
        model,
        dataset,
        optimizer,
        identifier,
        normalizer=None,
        timestamp_id: Optional[str] = None,
        run_dir: Optional[str] = None,
        is_debug: bool = False,
        is_hpo: bool = False,
        print_every: int = 100,
        seed: Optional[int] = None,
        logger: str = "tensorboard",
        local_rank: int = 0,
        amp: bool = False,
        cpu: bool = False,
        slurm={},
        noddp: bool = False,
    ) -> None:
        super().__init__(
            task=task,
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            identifier=identifier,
            normalizer=normalizer,
            timestamp_id=timestamp_id,
            run_dir=run_dir,
            is_debug=is_debug,
            is_hpo=is_hpo,
            print_every=print_every,
            seed=seed,
            logger=logger,
            local_rank=local_rank,
            amp=amp,
            cpu=cpu,
            name="is2rsv",
            slurm=slurm,
            noddp=noddp,
        )
        self.v_target=None

    def load_task(self) -> None:
        logging.info(f"Loading dataset: {self.config['task']['dataset']}")
        self.num_targets = 1

    # Takes in a new data source and generates predictions on it.
    @torch.no_grad()
    def predict(
        self,
        data_loader,
        per_image: bool = True,
        results_file=None,
        disable_tqdm: bool = False,
    ):
        ensure_fitted(self._unwrapped_model, warn=True)
        if distutils.is_master() and not disable_tqdm:
            logging.info("Predicting on test.")
        assert isinstance(
            data_loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = distutils.get_rank()

        if isinstance(data_loader, torch_geometric.data.Batch):
            data_loader = [[data_loader]]

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        predictions = {"id": [], "positions": []}
        for i, batch_list in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch_list)
            predictions['positions'].extend(out['positions'].cpu().detach())
            if per_image:
                if isinstance(batch_list[0].sid, list):
                    predictions["id"].extend(
                        [str(i) for i in batch_list[0].sid]
                    )
                else:
                    predictions["id"].extend(
                        [str(i) for i in batch_list[0].sid.tolist()]
                    )
            else:
                if isinstance(batch_list[0].sid, list):
                    predictions["id"].extend(
                        [str(i) for i in batch_list[0].sid]
                    )
                else:
                    predictions["id"].extend(
                        [str(i) for i in batch_list[0].sid.tolist()]
                    )
        self.save_results(
            predictions, results_file, keys=["positions"]
        )
        if self.ema:
            self.ema.restore()

        return predictions

    def update_best(
        self,
        primary_metric,
        val_metrics,
        disable_eval_tqdm: bool = True,
    ) -> None:
        if (
            "mae" in primary_metric
            and val_metrics[primary_metric]["metric"] < self.best_val_metric
        ) or (
            "mae" not in primary_metric
            and val_metrics[primary_metric]["metric"] > self.best_val_metric
        ):
            self.best_val_metric = val_metrics[primary_metric]["metric"]
            self.save(
                metrics=val_metrics,
                checkpoint_file="best_checkpoint.pt",
                training_state=False,
            )
            if self.test_loader is not None:
                self.predict(
                    self.test_loader,
                    results_file="predictions",
                    disable_tqdm=disable_eval_tqdm,
                    per_image=False
                )

    def train(self, disable_eval_tqdm: bool = False) -> None:

        ensure_fitted(self._unwrapped_model, warn=True)

        eval_every = self.config["optim"].get(
            "eval_every", len(self.train_loader)
        )
        checkpoint_every = self.config["optim"].get(
            "checkpoint_every", eval_every
        )
        primary_metric = self.config["task"].get(
            "primary_metric", self.evaluator.task_primary_metric['is2rs']
        )
        if (
            not hasattr(self, "primary_metric")
            or self.primary_metric != primary_metric
        ):
            self.best_val_metric = 1e9 if "mae" in primary_metric else -1.0
        else:
            primary_metric = self.primary_metric
        self.metrics = {}

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)

        for epoch_int in range(
            start_epoch, self.config["optim"]["max_epochs"]
        ):
            self.train_sampler.set_epoch(epoch_int)
            skip_steps = self.step % len(self.train_loader)
            train_loader_iter = iter(self.train_loader)

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()

                # Get a batch.
                batch = next(train_loader_iter)

                # is_back = True
                # Forward, loss, backward.
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    out = self._forward(batch)
                    loss = self._compute_loss(out, batch)

                loss = self.scaler.scale(loss) if self.scaler else loss

                self._backward(loss)
                scale = self.scaler.get_scale() if self.scaler else 1.0

                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out,
                    batch,
                    self.evaluator,
                    self.metrics,
                )
                self.metrics = self.evaluator.update(
                    "loss", loss.item() / scale, self.metrics
                )

                # Log metrics.
                log_dict = {k: self.metrics[k]["metric"] for k in self.metrics}
                log_dict.update(
                    {
                        "lr": self.scheduler.get_lr(),
                        "epoch": self.epoch,
                        "step": self.step,
                    }
                )
                if (
                    self.step % self.config["cmd"]["print_every"] == 0
                    and distutils.is_master()
                    and not self.is_hpo
                ):
                    log_str = [
                        "{}: {:.2e}".format(k, v) for k, v in log_dict.items()
                    ]
                    logging.info(", ".join(log_str))
                    self.metrics = {}

                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split="train",
                    )

                if (
                    checkpoint_every != -1
                    and self.step % checkpoint_every == 0
                ):
                    self.save(
                        checkpoint_file="checkpoint.pt", training_state=True
                    )

                # Evaluate on val set every `eval_every` iterations.
                if self.step % eval_every == 0:
                    if self.val_loader is not None:
                        val_metrics = self.validate(
                            split="val",
                            disable_tqdm=disable_eval_tqdm,
                        )
                        self.update_best(
                            primary_metric,
                            val_metrics,
                            disable_eval_tqdm=disable_eval_tqdm,
                        )
                        if self.is_hpo:
                            self.hpo_update(
                                self.epoch,
                                self.step,
                                self.metrics,
                                val_metrics,
                            )
                    if self.config["task"].get("eval_relaxations", False):
                        if "relax_dataset" not in self.config["task"]:
                            logging.warning(
                                "Cannot evaluate relaxations, relax_dataset not specified"
                            )
                        else:
                            self.run_relaxations()

                if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                    if self.step % eval_every == 0:
                        self.scheduler.step(
                            metrics=val_metrics[primary_metric]["metric"],
                        )
                else:
                    self.scheduler.step()

            torch.cuda.empty_cache()

            if checkpoint_every == -1:
                self.save(checkpoint_file="checkpoint.pt", training_state=True)

        self.train_dataset.close_db()
        if self.config.get("val_dataset", False):
            self.val_dataset.close_db()
        if self.config.get("test_dataset", False):
            self.test_dataset.close_db()

    def _forward(self, batch_list):
        # forward pass.
        # out = self.model(batch_list)
        out_v, out_p, main_graph = self.model(batch_list)
        out = {
            "vector": out_v if out_v is not None else torch.tensor([]),
            "positions": out_p if out_p is not None else torch.tensor([]),
        }
        batch = batch_list[0]
        if hasattr(batch, "pos_relaxed"):
            relax_graph = get_pbc_distances(
                batch.pos_relaxed,
                main_graph['edge_index'],
                batch.cell,
                -main_graph['cell_offset'],
                main_graph['num_neighbors'],
                return_distance_vec=True
            )
        if self.config["task"].get("train_on_free_atoms", True):
            # 如果使用完整结构，这里需要对边和原子进行筛选
            tags = batch.tags
            edge_index = main_graph['edge_index']
            src_tags = tags[edge_index[0]]
            dst_tags = tags[edge_index[1]]
            edge_mask = (src_tags != 0 ) & (dst_tags != 0)
            atom_mask = tags != 0
            out['vector'] = out['vector'][edge_mask]
            out['positions'] = out['positions'][atom_mask]
            self.v_target = -relax_graph['distance_vec'][edge_mask] # 因为main_graph在嵌入前对Vector取了反向，所以这里再次取反
        else:
            self.v_target = -relax_graph['distance_vec'] # 因为main_graph在嵌入前对Vector取了反向，所以这里再次取反
        return out

    def _compute_loss(self, out, batch_list) -> int:
        # ----------- 坐标损失
        pos_target = torch.cat([batch.pos_relaxed.to(self.device) for batch in batch_list])
        pos_origin = torch.cat([batch.pos.to(self.device) for batch in batch_list])
        p_mult = self.config["optim"].get("pos_coefficient", 1)
        if self.config["task"].get("train_on_free_atoms", True):
            tags = batch_list[0].tags
            atom_mask = tags != 0
            p_loss = p_mult*self.loss_fn['vector'](out['positions']+pos_origin[atom_mask],
                                                   pos_target[atom_mask])
        else:
            p_loss = p_mult*self.loss_fn['vector'](out['positions']+pos_origin,
                                                   pos_target)
        # ----------- 结构损失
        if out['vector'].shape[0]:
            v_target = self.v_target
            v_mult = self.config["optim"].get("graph_coefficient", 1)
            v_loss = v_mult * self.loss_fn["vector"](out["vector"],
                                                    v_target)
        else:
            v_loss = None
        # 先以相同权重生成loss
        loss = p_loss.clone()
        if v_loss is not None:
            loss = loss + v_loss
        return loss

    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        natoms = torch.cat(
            [batch.natoms.to(self.device) for batch in batch_list], dim=0
        )
        pos_origin = torch.cat([batch.pos.to(self.device) for batch in batch_list])
        p_target = torch.cat(
            [batch.pos_relaxed.to(self.device) for batch in batch_list], dim=0
        )
        if self.config["task"].get("train_on_free_atoms", True):
            tags = batch_list[0].tags
            atom_mask = tags != 0
            new_natoms = []
            start = 0
            for natom in natoms:
                end = start + natom
                # 对每个结构的原子进行 atom_mask 筛选
                mask = atom_mask[start:end]
                new_natoms.append(mask.sum())
                start = end

            natoms = torch.stack(new_natoms)
            out['positions'] = out['positions'] + pos_origin[atom_mask]
            p_target = p_target[atom_mask]
        else:
            out['positions'] = out['positions'] + pos_origin

        if out['vector'].shape[0]:
            g_target = self.v_target
        else:
            g_target = torch.tensor([])

        target = {
            "natoms": natoms,
            "vector": g_target,
            'positions':p_target,
            "pbc": torch.tensor([[True, True, False]]) * len(batch_list[0].cell),
            "cell": batch_list[0].cell,
        }
        out['natoms'] = natoms
        out['cell'] = batch_list[0].cell
        out['pbc'] = torch.tensor([[True, True, False]])*len(out['cell'])
        # print(out.keys())
        # print(target.keys())
        metrics = evaluator.eval(out, target, prev_metrics=metrics)
        return metrics
