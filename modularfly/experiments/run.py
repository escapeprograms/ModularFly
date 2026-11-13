"""Run one experiment into one run directory: train -> per-epoch Gini -> history.csv ->
checkpoint -> report. Experiment-agnostic: the caller builds the model (on its device)
and passes every config by name.

`configs` must hold "task" (TaskConfig), "train" (TrainConfig), "embedding"
(EmbeddingConfig), "interpretability" (InterpretabilityConfig) and "experiment"
(ExperimentConfig). Any other entries (data, dynamics, scale, interface, ...) are only
saved to config.json.
"""
import csv
import json
from collections.abc import Callable
from dataclasses import asdict, fields
from pathlib import Path
from typing import TextIO

import torch

from modularfly.config import ExperimentConfig, InterpretabilityConfig, TrainConfig
from modularfly.experiments.report import GINI_COLUMNS, gini_row, write_report
from modularfly.model.end_to_end.group_interface import GroupInterfaceModel
from modularfly.training.checkpoint import save_checkpoint
from modularfly.training.loop import EpochRecord, iter_epochs, make_optimizer
from modularfly.training.task import SplitData, make_split


def write_config(run_dir: Path, configs: dict[str, object], ecfg: ExperimentConfig) -> None:
    """run_dir / ecfg.config_name = {name: asdict(cfg)}; Paths and other non-JSON values as str."""
    payload = {name: asdict(cfg) for name, cfg in configs.items()}
    (run_dir / ecfg.config_name).write_text(json.dumps(payload, default=str, indent=2))


def open_history(run_dir: Path, ecfg: ExperimentConfig) -> tuple[TextIO, csv.DictWriter]:
    """Opens run_dir / ecfg.history_name and writes the header (EpochRecord fields + GINI_COLUMNS).
    The caller closes the file."""
    file = open(run_dir / ecfg.history_name, "w", newline="")
    columns = [f.name for f in fields(EpochRecord)] + list(GINI_COLUMNS)
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    file.flush()
    return file, writer


def log_epoch(record: EpochRecord, model: GroupInterfaceModel, icfg: InterpretabilityConfig,
              file: TextIO, writer: csv.DictWriter) -> dict:
    """The history row of one epoch (record + Gini), written and flushed so a killed run keeps it."""
    row = asdict(record) | gini_row(model, icfg)
    writer.writerow(row)
    file.flush()
    return row


def train_and_log(model: GroupInterfaceModel, data: SplitData, configs: dict[str, object],
                  run_dir: Path, on_row: Callable[[dict], None]) -> list[dict]:
    """Trains with iter_epochs; after each epoch logs the row, calls on_row and overwrites
    the checkpoint. Returns the rows."""
    train_cfg: TrainConfig = configs["train"]
    icfg, ecfg = configs["interpretability"], configs["experiment"]
    optimizer = make_optimizer(model, train_cfg)
    generator = torch.Generator().manual_seed(train_cfg.seed)
    rows = []
    file, writer = open_history(run_dir, ecfg)
    with file:
        for record in iter_epochs(model, optimizer, data, train_cfg, generator):
            rows.append(log_epoch(record, model, icfg, file, writer))
            on_row(rows[-1])
            save_checkpoint(run_dir / ecfg.checkpoint_name, model, optimizer, record.epoch)
    return rows


def run_experiment(model: GroupInterfaceModel, configs: dict[str, object], run_dir: Path,
                   on_row: Callable[[dict], None]) -> list[dict]:
    """Creates run_dir (raises FileExistsError if it exists), saves the configs, trains on the
    split of configs["task"] on the model's device, then writes the report. Returns the rows."""
    run_dir.mkdir(parents=True, exist_ok=False)
    write_config(run_dir, configs, configs["experiment"])
    device = next(model.parameters()).device
    data = make_split(configs["embedding"].modulus, configs["task"]).to(device)
    rows = train_and_log(model, data, configs, run_dir, on_row)
    write_report(run_dir, model, rows, configs["interpretability"], configs["experiment"])
    return rows
