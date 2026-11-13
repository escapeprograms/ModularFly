import csv
import json

import pytest

from modularfly.config import (
    ExperimentConfig, InterpretabilityConfig, TaskConfig, TrainConfig,
)
from modularfly.experiments.report import GINI_COLUMNS
from modularfly.experiments.run import run_experiment
from modularfly.training.checkpoint import load_checkpoint
from tests.conftest import SMALL_EMB, make_model, trainable_fly


def small_configs(tmp_path):
    return {
        "task": TaskConfig(train_fraction=0.5),
        "train": TrainConfig(batch_size=8, n_epochs=2, eval_batch=16),
        "embedding": SMALL_EMB,
        "interpretability": InterpretabilityConfig(),
        "experiment": ExperimentConfig(device="cpu", runs_dir=tmp_path),
    }


@pytest.fixture
def finished_run(small_graph, tmp_path):
    configs = small_configs(tmp_path)
    run_dir = tmp_path / "run"
    seen = []
    rows = run_experiment(make_model(small_graph, trainable_fly(small_graph)), configs,
                          run_dir, on_row=seen.append)
    return run_dir, configs["experiment"], rows, seen


def test_config_json_has_every_config(finished_run):
    run_dir, ecfg, _, _ = finished_run
    saved = json.loads((run_dir / ecfg.config_name).read_text())
    assert set(saved) == {"task", "train", "embedding", "interpretability", "experiment"}
    assert saved["train"]["n_epochs"] == 2 and saved["experiment"]["device"] == "cpu"


def test_history_has_one_row_per_epoch_with_gini(finished_run):
    run_dir, ecfg, rows, seen = finished_run
    with open(run_dir / ecfg.history_name, newline="") as file:
        history = list(csv.DictReader(file))
    assert [int(r["epoch"]) for r in history] == [0, 1]
    assert all(col in history[0] for col in GINI_COLUMNS)
    assert "theta_drift" in history[0]
    assert seen == rows and len(rows) == 2


def test_checkpoint_and_report_written(finished_run):
    run_dir, ecfg, _, _ = finished_run
    assert load_checkpoint(run_dir / ecfg.checkpoint_name)["epoch"] == 1
    assert (run_dir / ecfg.report_name).exists()
    assert len(list((run_dir / ecfg.figure_dir_name).glob("*.png"))) == 6


def test_existing_run_dir_raises(small_graph, tmp_path):
    (tmp_path / "run").mkdir()
    model = make_model(small_graph, trainable_fly(small_graph))
    with pytest.raises(FileExistsError):
        run_experiment(model, small_configs(tmp_path), tmp_path / "run", on_row=print)
