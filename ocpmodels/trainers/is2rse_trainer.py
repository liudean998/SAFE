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

@registry.register_trainer("is2rse")
class Is2RseTrainer(BaseTrainer):
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
            name="is2rfse",
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
        if self.normalizers is not None:
            if "target" in self.normalizers:
                self.normalizers["target"].to(self.device)

        # predictions = {"id": [], "energy": [], "vector":[]}
        predictions = {"id": [], "energy": []}
        for i, batch_list in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch_list)
            if self.normalizers is not None :
                if "target" in self.normalizers:
                    out["energy"] = self.normalizers["target"].denorm(
                        out["energy"]
                    )
            predictions['energy'].extend(out['energy'].cpu().detach())
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

        # predictions["id"] = np.array(
        #     predictions["id"],
        # )
        self.save_results(
            predictions, results_file, keys=["energy"]
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
            "primary_metric", self.evaluator.task_primary_metric['is2rve']
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
                    total_norm = 0
                    # for name, param in self.model.named_parameters():
                    #     if param.grad is not None:
                    #         grad_norm = param.grad.data.norm(2).item()
                    #         print(f'{name}: grad norm = {grad_norm:.6f}')
                    #         total_norm += grad_norm
                    # print(f'Total grad norm = {total_norm:.6f}')
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
        out_e, out_v, out_p, out_f, main_graph = self.model(batch_list)
        out = {
            'energy': out_e,
            "vector": out_v if out_v is not None else torch.tensor([]),
            "positions": out_p if out_p is not None else torch.tensor([]),
            "forces": out_f if out_f is not None else torch.tensor([]),
        }
        batch = batch_list[0]
        # # 如果使用完整结构，这里需要对边和原子进行筛选
        # tags = batch.tags
        # edge_index = main_graph['edge_index']
        # src_tags = tags[edge_index[0]]
        # dst_tags = tags[edge_index[1]]
        # edge_mask = (src_tags == 0 ) & (dst_tags == 0)
        # atom_mask = tags != 0
        # out['vector'] = out['vector'][edge_mask]
        # out['positions'] = out['positions'][atom_mask]
        if hasattr(batch, "pos_relaxed"):
            relax_graph = get_pbc_distances(
                batch.pos_relaxed,
                main_graph['edge_index'],
                batch.cell,
                -main_graph['cell_offset'],
                main_graph['num_neighbors'],
                return_distance_vec=True
            )
            # self.v_target = -relax_graph['distance_vec'][edge_mask] # 因为main_graph在嵌入前对Vector取了反向，所以这里再次取反
            self.v_target = -relax_graph['distance_vec'] # 因为main_graph在嵌入前对Vector取了反向，所以这里再次取反
        return out

    def _compute_loss(self, out, batch_list) -> int:
        # ----------- 能量损失
        energy_target = torch.cat(
            [batch.y.to(self.device) for batch in batch_list], dim=0
        )
        if self.normalizer.get("normalize_labels", False):
            if "target" in self.normalizers:
                target_normed = self.normalizers["target"].norm(energy_target)
            else:
                target_normed = energy_target
        else:
            target_normed = energy_target
        e_mult = self.config["optim"].get("energy_coefficient", 1)
        energy_loss = e_mult*self.loss_fn["energy"](out["energy"], target_normed)
        # ----------- 结构损失
        if out['vector'].shape[0]:
            v_target = self.v_target
            v_mult = self.config["optim"].get("graph_coefficient", 1)
            v_loss = v_mult * self.loss_fn["vector"](out["vector"],
                                                    v_target)
        else:
            v_loss = None

        # ----------- 坐标损失
        if out['positions'].shape[0]:
            # tags = batch_list[0].tags
            # atom_mask = tags != 0
            pos_target = torch.cat([batch.pos_relaxed.to(self.device) for batch in batch_list])
            pos_origin = torch.cat([batch.pos.to(self.device) for batch in batch_list])
            p_mult = self.config["optim"].get("pos_coefficient", 1)
            # p_loss = p_mult*self.loss_fn['vector'](out['positions']+pos_origin[atom_mask],
            #                                        pos_target[atom_mask])
            p_loss = p_mult*self.loss_fn['vector'](out['positions']+pos_origin,
                                                   pos_target)
        else:
            p_loss = None

        # ----------- 力损失
        if out['forces'].shape[0]:
            force_target = torch.cat([batch.force.to(self.device) for batch in batch_list])
            f_mult = self.config["optim"].get("force_coefficient", 1)
            f_loss = f_mult*self.loss_fn['force'](out['forces'],
                                                   force_target)
        else:
            f_loss = None

        # 先以相同权重生成loss
        loss = energy_loss.clone()
        if v_loss is not None:
            loss = loss + v_loss
        if p_loss is not None:
            loss = loss + p_loss
        if f_loss is not  None:
            loss = loss + f_loss
        return loss

    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        natoms = torch.cat(
            [batch.natoms.to(self.device) for batch in batch_list], dim=0
        )
        e_target = torch.cat(
            [batch.y.to(self.device) for batch in batch_list], dim=0
        )
        if out['vector'].shape[0]:
            g_target = self.v_target
        else:
            g_target = torch.tensor([])

        if out['positions'].shape[0]:
            # tags = batch_list[0].tags
            # atom_mask = tags != 0
            pos_origin = torch.cat([batch.pos.to(self.device) for batch in batch_list])
            # out['positions'] = out['positions'] + pos_origin[atom_mask]
            out['positions'] = out['positions'] + pos_origin
            p_target = torch.cat(
                [batch.pos_relaxed.to(self.device) for batch in batch_list], dim=0
            )
            # p_target = p_target[atom_mask]
        else:
            p_target = torch.tensor([])

        if out['forces'].shape[0]:
            f_target = torch.cat([batch.force.to(self.device) for batch in batch_list])
        else:
            f_target = torch.tensor([])

        target = {
            "natoms": natoms,
            "vector": g_target,
            'energy': e_target,
            'positions':p_target,
            'forces': f_target
        }
        out['natoms'] = natoms

        if self.normalizer.get("normalize_labels", False):
            if 'target' in self.normalizers:
                out["energy"] = self.normalizers["target"].denorm(out["energy"])
        metrics = evaluator.eval(out, target, prev_metrics=metrics)
        return metrics

    def run_relaxations(self, split: str = "val") -> None:
        ensure_fitted(self._unwrapped_model)

        # When set to true, uses deterministic CUDA scatter ops, if available.
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
        # Only implemented for GemNet-OC currently.
        registry.register(
            "set_deterministic_scatter",
            self.config["task"].get("set_deterministic_scatter", False),
        )

        logging.info("Running ML-relaxations")
        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        evaluator_is2rs, metrics_is2rs = Evaluator(task="is2rs"), {}
        evaluator_is2re, metrics_is2re = Evaluator(task="is2re"), {}

        # Need both `pos_relaxed` and `y_relaxed` to compute val IS2R* metrics.
        # Else just generate predictions.
        if (
            hasattr(self.relax_dataset[0], "pos_relaxed")
            and self.relax_dataset[0].pos_relaxed is not None
        ) and (
            hasattr(self.relax_dataset[0], "y_relaxed")
            and self.relax_dataset[0].y_relaxed is not None
        ):
            split = "val"
        else:
            split = "test"

        ids = []
        relaxed_positions = []
        chunk_idx = []
        for i, batch in tqdm(
            enumerate(self.relax_loader), total=len(self.relax_loader)
        ):
            if i >= self.config["task"].get("num_relaxation_batches", 1e9):
                break

            # If all traj files already exist, then skip this batch
            if check_traj_files(
                batch, self.config["task"]["relax_opt"].get("traj_dir", None)
            ):
                logging.info(f"Skipping batch: {batch[0].sid.tolist()}")
                continue

            relaxed_batch = ml_relax(
                batch=batch,
                model=self,
                steps=self.config["task"].get("relaxation_steps", 200),
                fmax=self.config["task"].get("relaxation_fmax", 0.0),
                relax_opt=self.config["task"]["relax_opt"],
                save_full_traj=self.config["task"].get("save_full_traj", True),
                device=self.device,
                transform=None,
            )

            if self.config["task"].get("write_pos", False):
                systemids = [str(i) for i in relaxed_batch.sid.tolist()]
                natoms = relaxed_batch.natoms.tolist()
                positions = torch.split(relaxed_batch.pos, natoms)
                batch_relaxed_positions = [pos.tolist() for pos in positions]

                relaxed_positions += batch_relaxed_positions
                chunk_idx += natoms
                ids += systemids

            if split == "val":
                mask = relaxed_batch.fixed == 0
                s_idx = 0
                natoms_free = []
                for natoms in relaxed_batch.natoms:
                    natoms_free.append(
                        torch.sum(mask[s_idx : s_idx + natoms]).item()
                    )
                    s_idx += natoms

                target = {
                    "energy": relaxed_batch.y_relaxed,
                    "positions": relaxed_batch.pos_relaxed[mask],
                    "cell": relaxed_batch.cell,
                    "pbc": torch.tensor([True, True, True]),
                    "natoms": torch.LongTensor(natoms_free),
                }

                prediction = {
                    "energy": relaxed_batch.y,
                    "positions": relaxed_batch.pos[mask],
                    "cell": relaxed_batch.cell,
                    "pbc": torch.tensor([True, True, True]),
                    "natoms": torch.LongTensor(natoms_free),
                }

                metrics_is2rs = evaluator_is2rs.eval(
                    prediction,
                    target,
                    metrics_is2rs,
                )
                metrics_is2re = evaluator_is2re.eval(
                    {"energy": prediction["energy"]},
                    {"energy": target["energy"]},
                    metrics_is2re,
                )

        if self.config["task"].get("write_pos", False):
            rank = distutils.get_rank()
            pos_filename = os.path.join(
                self.config["cmd"]["results_dir"], f"relaxed_pos_{rank}.npz"
            )
            np.savez_compressed(
                pos_filename,
                ids=ids,
                pos=np.array(relaxed_positions, dtype=object),
                chunk_idx=chunk_idx,
            )

            distutils.synchronize()
            if distutils.is_master():
                gather_results = defaultdict(list)
                full_path = os.path.join(
                    self.config["cmd"]["results_dir"],
                    "relaxed_positions.npz",
                )

                for i in range(distutils.get_world_size()):
                    rank_path = os.path.join(
                        self.config["cmd"]["results_dir"],
                        f"relaxed_pos_{i}.npz",
                    )
                    rank_results = np.load(rank_path, allow_pickle=True)
                    gather_results["ids"].extend(rank_results["ids"])
                    gather_results["pos"].extend(rank_results["pos"])
                    gather_results["chunk_idx"].extend(
                        rank_results["chunk_idx"]
                    )
                    os.remove(rank_path)

                # Because of how distributed sampler works, some system ids
                # might be repeated to make no. of samples even across GPUs.
                _, idx = np.unique(gather_results["ids"], return_index=True)
                gather_results["ids"] = np.array(gather_results["ids"])[idx]
                gather_results["pos"] = np.concatenate(
                    np.array(gather_results["pos"])[idx]
                )
                gather_results["chunk_idx"] = np.cumsum(
                    np.array(gather_results["chunk_idx"])[idx]
                )[
                    :-1
                ]  # np.split does not need last idx, assumes n-1:end

                logging.info(f"Writing results to {full_path}")
                np.savez_compressed(full_path, **gather_results)

        if split == "val":
            for task in ["is2rs", "is2re"]:
                metrics = eval(f"metrics_{task}")
                aggregated_metrics = {}
                for k in metrics:
                    aggregated_metrics[k] = {
                        "total": distutils.all_reduce(
                            metrics[k]["total"],
                            average=False,
                            device=self.device,
                        ),
                        "numel": distutils.all_reduce(
                            metrics[k]["numel"],
                            average=False,
                            device=self.device,
                        ),
                    }
                    aggregated_metrics[k]["metric"] = (
                        aggregated_metrics[k]["total"]
                        / aggregated_metrics[k]["numel"]
                    )
                metrics = aggregated_metrics

                # Make plots.
                log_dict = {
                    f"{task}_{k}": metrics[k]["metric"] for k in metrics
                }
                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split=split,
                    )

                if distutils.is_master():
                    logging.info(metrics)

        if self.ema:
            self.ema.restore()

        registry.unregister("set_deterministic_scatter")

@registry.register_trainer("is2rse_abs")
class Is2RseTrainer(BaseTrainer):
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
            name="is2rfse",
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
        if self.normalizers is not None:
            if "target" in self.normalizers:
                self.normalizers["target"].to(self.device)

        # predictions = {"id": [], "energy": [], "vector":[]}
        predictions = {"id": [], "energy": []}
        for i, batch_list in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch_list)
            if self.normalizers is not None :
                if "target" in self.normalizers:
                    out["energy"] = self.normalizers["target"].denorm(
                        out["energy"]
                    )
            predictions['energy'].extend(out['energy'].cpu().detach())
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

        # predictions["id"] = np.array(
        #     predictions["id"],
        # )
        self.save_results(
            predictions, results_file, keys=["energy"]
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
            "primary_metric", self.evaluator.task_primary_metric['is2rve']
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
                    total_norm = 0
                    # for name, param in self.model.named_parameters():
                    #     if param.grad is not None:
                    #         grad_norm = param.grad.data.norm(2).item()
                    #         print(f'{name}: grad norm = {grad_norm:.6f}')
                    #         total_norm += grad_norm
                    # print(f'Total grad norm = {total_norm:.6f}')
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
        out_e, out_v, out_p, out_f, main_graph = self.model(batch_list)
        # print(out_v)
        out = {
            'energy': out_e,
            "vector": out_v if out_v is not None else torch.tensor([]),
            "positions": out_p if out_p is not None else torch.tensor([]),
            "forces": out_f if out_f is not None else torch.tensor([]),
        }
        batch = batch_list[0]
        # # 如果使用完整结构，这里需要对边和原子进行筛选
        # tags = batch.tags
        # edge_index = main_graph['edge_index']
        # src_tags = tags[edge_index[0]]
        # dst_tags = tags[edge_index[1]]
        # edge_mask = (src_tags == 0 ) & (dst_tags == 0)
        # atom_mask = tags != 0
        # out['vector'] = out['vector'][edge_mask]
        # out['positions'] = out['positions'][atom_mask]
        if hasattr(batch, "pos_relaxed"):
            relax_graph = get_pbc_distances(
                batch.pos_relaxed,
                main_graph['edge_index'],
                batch.cell,
                -main_graph['cell_offset'],
                main_graph['num_neighbors'],
                return_distance_vec=True
            )
            # self.v_target = -relax_graph['distance_vec'][edge_mask] # 因为main_graph在嵌入前对Vector取了反向，所以这里再次取反
            self.v_target = torch.abs(relax_graph['distance_vec']) # 因为main_graph在嵌入前对Vector取了反向，所以这里再次取反
            # print(self.v_target)
            # import time
            # time.sleep(1000)
        return out

    def _compute_loss(self, out, batch_list) -> int:
        # ----------- 能量损失
        energy_target = torch.cat(
            [batch.y.to(self.device) for batch in batch_list], dim=0
        )
        if self.normalizer.get("normalize_labels", False):
            if "target" in self.normalizers:
                target_normed = self.normalizers["target"].norm(energy_target)
            else:
                target_normed = energy_target
        else:
            target_normed = energy_target
        e_mult = self.config["optim"].get("energy_coefficient", 1)
        energy_loss = e_mult*self.loss_fn["energy"](out["energy"], target_normed)
        # ----------- 结构损失
        if out['vector'].shape[0]:
            v_target = self.v_target
            v_mult = self.config["optim"].get("graph_coefficient", 1)
            v_loss = v_mult * self.loss_fn["vector"](out["vector"],
                                                    v_target)
        else:
            v_loss = None

        # ----------- 坐标损失
        if out['positions'].shape[0]:
            # tags = batch_list[0].tags
            # atom_mask = tags != 0
            pos_target = torch.cat([batch.pos_relaxed.to(self.device) for batch in batch_list])
            pos_origin = torch.cat([batch.pos.to(self.device) for batch in batch_list])
            p_mult = self.config["optim"].get("pos_coefficient", 1)
            # p_loss = p_mult*self.loss_fn['vector'](out['positions']+pos_origin[atom_mask],
            #                                        pos_target[atom_mask])
            p_loss = p_mult*self.loss_fn['vector'](out['positions']+pos_origin,
                                                   pos_target)
        else:
            p_loss = None

        # ----------- 力损失
        if out['forces'].shape[0]:
            force_target = torch.cat([batch.force.to(self.device) for batch in batch_list])
            f_mult = self.config["optim"].get("force_coefficient", 1)
            f_loss = f_mult*self.loss_fn['force'](out['forces'],
                                                   force_target)
        else:
            f_loss = None

        # 先以相同权重生成loss
        loss = energy_loss.clone()
        if v_loss is not None:
            loss = loss + v_loss
        if p_loss is not None:
            loss = loss + p_loss
        if f_loss is not  None:
            loss = loss + f_loss
        return loss

    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        natoms = torch.cat(
            [batch.natoms.to(self.device) for batch in batch_list], dim=0
        )
        e_target = torch.cat(
            [batch.y.to(self.device) for batch in batch_list], dim=0
        )
        if out['vector'].shape[0]:
            g_target = self.v_target
        else:
            g_target = torch.tensor([])

        if out['positions'].shape[0]:
            # tags = batch_list[0].tags
            # atom_mask = tags != 0
            pos_origin = torch.cat([batch.pos.to(self.device) for batch in batch_list])
            # out['positions'] = out['positions'] + pos_origin[atom_mask]
            out['positions'] = out['positions'] + pos_origin
            p_target = torch.cat(
                [batch.pos_relaxed.to(self.device) for batch in batch_list], dim=0
            )
            # p_target = p_target[atom_mask]
        else:
            p_target = torch.tensor([])

        if out['forces'].shape[0]:
            f_target = torch.cat([batch.force.to(self.device) for batch in batch_list])
        else:
            f_target = torch.tensor([])

        target = {
            "natoms": natoms,
            "vector": g_target,
            'energy': e_target,
            'positions':p_target,
            'forces': f_target
        }
        out['natoms'] = natoms

        if self.normalizer.get("normalize_labels", False):
            if 'target' in self.normalizers:
                out["energy"] = self.normalizers["target"].denorm(out["energy"])
        metrics = evaluator.eval(out, target, prev_metrics=metrics)
        return metrics

    def run_relaxations(self, split: str = "val") -> None:
        ensure_fitted(self._unwrapped_model)

        # When set to true, uses deterministic CUDA scatter ops, if available.
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
        # Only implemented for GemNet-OC currently.
        registry.register(
            "set_deterministic_scatter",
            self.config["task"].get("set_deterministic_scatter", False),
        )

        logging.info("Running ML-relaxations")
        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        evaluator_is2rs, metrics_is2rs = Evaluator(task="is2rs"), {}
        evaluator_is2re, metrics_is2re = Evaluator(task="is2re"), {}

        # Need both `pos_relaxed` and `y_relaxed` to compute val IS2R* metrics.
        # Else just generate predictions.
        if (
            hasattr(self.relax_dataset[0], "pos_relaxed")
            and self.relax_dataset[0].pos_relaxed is not None
        ) and (
            hasattr(self.relax_dataset[0], "y_relaxed")
            and self.relax_dataset[0].y_relaxed is not None
        ):
            split = "val"
        else:
            split = "test"

        ids = []
        relaxed_positions = []
        chunk_idx = []
        for i, batch in tqdm(
            enumerate(self.relax_loader), total=len(self.relax_loader)
        ):
            if i >= self.config["task"].get("num_relaxation_batches", 1e9):
                break

            # If all traj files already exist, then skip this batch
            if check_traj_files(
                batch, self.config["task"]["relax_opt"].get("traj_dir", None)
            ):
                logging.info(f"Skipping batch: {batch[0].sid.tolist()}")
                continue

            relaxed_batch = ml_relax(
                batch=batch,
                model=self,
                steps=self.config["task"].get("relaxation_steps", 200),
                fmax=self.config["task"].get("relaxation_fmax", 0.0),
                relax_opt=self.config["task"]["relax_opt"],
                save_full_traj=self.config["task"].get("save_full_traj", True),
                device=self.device,
                transform=None,
            )

            if self.config["task"].get("write_pos", False):
                systemids = [str(i) for i in relaxed_batch.sid.tolist()]
                natoms = relaxed_batch.natoms.tolist()
                positions = torch.split(relaxed_batch.pos, natoms)
                batch_relaxed_positions = [pos.tolist() for pos in positions]

                relaxed_positions += batch_relaxed_positions
                chunk_idx += natoms
                ids += systemids

            if split == "val":
                mask = relaxed_batch.fixed == 0
                s_idx = 0
                natoms_free = []
                for natoms in relaxed_batch.natoms:
                    natoms_free.append(
                        torch.sum(mask[s_idx : s_idx + natoms]).item()
                    )
                    s_idx += natoms

                target = {
                    "energy": relaxed_batch.y_relaxed,
                    "positions": relaxed_batch.pos_relaxed[mask],
                    "cell": relaxed_batch.cell,
                    "pbc": torch.tensor([True, True, True]),
                    "natoms": torch.LongTensor(natoms_free),
                }

                prediction = {
                    "energy": relaxed_batch.y,
                    "positions": relaxed_batch.pos[mask],
                    "cell": relaxed_batch.cell,
                    "pbc": torch.tensor([True, True, True]),
                    "natoms": torch.LongTensor(natoms_free),
                }

                metrics_is2rs = evaluator_is2rs.eval(
                    prediction,
                    target,
                    metrics_is2rs,
                )
                metrics_is2re = evaluator_is2re.eval(
                    {"energy": prediction["energy"]},
                    {"energy": target["energy"]},
                    metrics_is2re,
                )

        if self.config["task"].get("write_pos", False):
            rank = distutils.get_rank()
            pos_filename = os.path.join(
                self.config["cmd"]["results_dir"], f"relaxed_pos_{rank}.npz"
            )
            np.savez_compressed(
                pos_filename,
                ids=ids,
                pos=np.array(relaxed_positions, dtype=object),
                chunk_idx=chunk_idx,
            )

            distutils.synchronize()
            if distutils.is_master():
                gather_results = defaultdict(list)
                full_path = os.path.join(
                    self.config["cmd"]["results_dir"],
                    "relaxed_positions.npz",
                )

                for i in range(distutils.get_world_size()):
                    rank_path = os.path.join(
                        self.config["cmd"]["results_dir"],
                        f"relaxed_pos_{i}.npz",
                    )
                    rank_results = np.load(rank_path, allow_pickle=True)
                    gather_results["ids"].extend(rank_results["ids"])
                    gather_results["pos"].extend(rank_results["pos"])
                    gather_results["chunk_idx"].extend(
                        rank_results["chunk_idx"]
                    )
                    os.remove(rank_path)

                # Because of how distributed sampler works, some system ids
                # might be repeated to make no. of samples even across GPUs.
                _, idx = np.unique(gather_results["ids"], return_index=True)
                gather_results["ids"] = np.array(gather_results["ids"])[idx]
                gather_results["pos"] = np.concatenate(
                    np.array(gather_results["pos"])[idx]
                )
                gather_results["chunk_idx"] = np.cumsum(
                    np.array(gather_results["chunk_idx"])[idx]
                )[
                    :-1
                ]  # np.split does not need last idx, assumes n-1:end

                logging.info(f"Writing results to {full_path}")
                np.savez_compressed(full_path, **gather_results)

        if split == "val":
            for task in ["is2rs", "is2re"]:
                metrics = eval(f"metrics_{task}")
                aggregated_metrics = {}
                for k in metrics:
                    aggregated_metrics[k] = {
                        "total": distutils.all_reduce(
                            metrics[k]["total"],
                            average=False,
                            device=self.device,
                        ),
                        "numel": distutils.all_reduce(
                            metrics[k]["numel"],
                            average=False,
                            device=self.device,
                        ),
                    }
                    aggregated_metrics[k]["metric"] = (
                        aggregated_metrics[k]["total"]
                        / aggregated_metrics[k]["numel"]
                    )
                metrics = aggregated_metrics

                # Make plots.
                log_dict = {
                    f"{task}_{k}": metrics[k]["metric"] for k in metrics
                }
                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split=split,
                    )

                if distutils.is_master():
                    logging.info(metrics)

        if self.ema:
            self.ema.restore()

        registry.unregister("set_deterministic_scatter")
