"""W&B logging in disabled mode with a dummy key in a temporary .env (never the real file)."""
import socket

import pytest

from modularfly.config import WandbConfig
from modularfly.experiments.report import spectra_summary
from modularfly.experiments.run import run_experiment
from modularfly.experiments.wandb_logging import finish_run, log_row, read_env_value, start_run
from tests.conftest import make_model, trainable_fly
from tests.test_run_experiment import small_configs


def write_env(tmp_path, text: str):
    path = tmp_path / ".env"
    path.write_text(text)
    return path


def test_read_env_value(tmp_path):
    assert read_env_value(write_env(tmp_path, "WANDB_KEY=dummy\n"), "WANDB_KEY") == "dummy"


def test_read_env_value_missing_name_raises(tmp_path):
    with pytest.raises(KeyError):
        read_env_value(write_env(tmp_path, "OTHER=x\n"), "WANDB_KEY")


def test_read_env_value_empty_raises(tmp_path):
    with pytest.raises(KeyError):
        read_env_value(write_env(tmp_path, "WANDB_KEY=\n"), "WANDB_KEY")


LOOPBACK = ("127.0.0.1", "::1", "localhost")
REAL_CONNECT = socket.socket.connect


def loopback_only_connect(sock, address):
    """socket.connect that allows loopback (asyncio's self-pipe on Windows) and fails otherwise."""
    if not (isinstance(address, tuple) and address[0] in LOOPBACK):
        raise AssertionError(f"disabled W&B connected to {address}")
    return REAL_CONNECT(sock, address)


def test_disabled_run_logs_without_network(small_graph, tmp_path, monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", loopback_only_connect)
    cfg = WandbConfig(mode="disabled", env_file=write_env(tmp_path, "WANDB_KEY=dummy\n"))
    configs = small_configs(tmp_path) | {"wandb": cfg}
    model = make_model(small_graph, trainable_fly(small_graph))
    run = start_run(cfg, "test", configs)
    run_dir = tmp_path / "run"
    rows = run_experiment(model, configs, run_dir, on_row=lambda row: log_row(run, row))
    assert len(rows) == 2
    summary = spectra_summary(model, configs["interpretability"])
    finish_run(run, run_dir, summary, cfg, configs["experiment"])
