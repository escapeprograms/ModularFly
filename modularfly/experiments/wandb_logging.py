"""Weights & Biases monitoring of one experiment run. The only module that imports
`wandb` or `dotenv`; run.py and report.py know nothing about W&B.

The hook is `run_experiment`'s `on_row` callback: the script starts a run with
`start_run`, passes an `on_row` that prints the row and calls `log_row`, and after
`run_experiment` returns calls `finish_run` with the run directory and `spectra_summary`.

The API key is read from `WandbConfig.env_file` and handed to `wandb.Settings`, so
nothing is written to the user's netrc and no environment variable is set.
"""
import json
from dataclasses import asdict
from pathlib import Path

import wandb
from dotenv import dotenv_values

from modularfly.config import ExperimentConfig, WandbConfig
from modularfly.experiments.report import TABLE_NAMES


def read_env_value(path: Path, name: str) -> str:
    """The value of `name` in the dotenv file at `path`; KeyError if it is missing or empty."""
    value = dotenv_values(path).get(name)
    if not value:
        raise KeyError(f"{name} is missing or empty in {path}")
    return value


def wandb_settings(cfg: WandbConfig) -> wandb.Settings:
    """Settings carrying the API key from cfg.env_file and cfg.mode."""
    return wandb.Settings(api_key=read_env_value(cfg.env_file, cfg.key_variable), mode=cfg.mode)


def json_safe_configs(configs: dict[str, object]) -> dict:
    """{name: asdict(cfg)} with Paths and other non-JSON values as str (as in config.json)."""
    return json.loads(json.dumps({name: asdict(c) for name, c in configs.items()}, default=str))


def start_run(cfg: WandbConfig, run_name: str, configs: dict[str, object]) -> wandb.Run:
    """A new W&B run named run_name in cfg.entity/cfg.project, with every config attached."""
    return wandb.init(entity=cfg.entity, project=cfg.project, name=run_name,
                      config=json_safe_configs(configs), settings=wandb_settings(cfg))


def log_row(run: wandb.Run, row: dict) -> None:
    """One history row at W&B step row["epoch"]."""
    run.log(row, step=row["epoch"])


def log_figures(run: wandb.Run, figure_dir: Path) -> None:
    """Every PNG in figure_dir as figures/<stem>, in one call without a step: W&B adds it
    to the current (last logged) step, so steps never go backwards."""
    run.log({f"figures/{png.stem}": wandb.Image(str(png)) for png in sorted(figure_dir.glob("*.png"))})


def log_summary(run: wandb.Run, summary: dict) -> None:
    """<table>/gini and <table>/dominant of spectra_summary into the run summary."""
    for table in TABLE_NAMES:
        run.summary[f"{table}/gini"] = summary[table]["gini"]
        run.summary[f"{table}/dominant"] = summary[table]["dominant"]


def log_run_files(run: wandb.Run, run_dir: Path, cfg: WandbConfig, ecfg: ExperimentConfig) -> None:
    """config.json, history.csv, the report and figures/ as one artifact. The checkpoint stays local."""
    artifact = wandb.Artifact(cfg.artifact_name, type="run")
    for name in (ecfg.config_name, ecfg.history_name, ecfg.report_name):
        artifact.add_file(str(run_dir / name))
    artifact.add_dir(str(run_dir / ecfg.figure_dir_name), name=ecfg.figure_dir_name)
    run.log_artifact(artifact)


def finish_run(run: wandb.Run, run_dir: Path, summary: dict, cfg: WandbConfig,
               ecfg: ExperimentConfig) -> None:
    """Logs the figures, the summary and the run files, then finishes the run."""
    log_figures(run, run_dir / ecfg.figure_dir_name)
    log_summary(run, summary)
    log_run_files(run, run_dir, cfg, ecfg)
    run.finish()
