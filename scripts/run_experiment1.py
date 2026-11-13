"""Experiment 1 (project_design.md): train random-init fly weights + token embeddings on
modular addition through a sampled channel-group interface (arXiv 2610.10014), then report
the embedding spectrum."""
import argparse
import time
from dataclasses import replace
from functools import partial

from modularfly.config import (
    DataConfig, DynamicsConfig, EmbeddingConfig, ExperimentConfig, InterfaceConfig,
    InterpretabilityConfig, ScaleConfig, TaskConfig, TrainConfig, WandbConfig,
)
from modularfly.data.graph import graph_cache_path, load_graph
from modularfly.experiments.fly_model import build_group_interface_model, random_trainable_weights
from modularfly.experiments.report import spectra_summary
from modularfly.experiments.run import run_experiment
from modularfly.experiments.wandb_logging import finish_run, log_row, start_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--eval-batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--fly-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--fly-weight-decay", type=float, default=None)
    parser.add_argument("--fly-penalty", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    return parser.parse_args()


def override(cfg, **values):
    """A copy of the dataclass cfg with every non-None value applied."""
    return replace(cfg, **{name: v for name, v in values.items() if v is not None})


def build_configs(args: argparse.Namespace) -> dict[str, object]:
    train = override(TrainConfig(), n_epochs=args.epochs, batch_size=args.batch,
                     eval_batch=args.eval_batch, lr=args.lr, fly_lr=args.fly_lr,
                     weight_decay=args.weight_decay, fly_weight_decay=args.fly_weight_decay,
                     fly_penalty=args.fly_penalty, grad_clip_norm=args.grad_clip_norm,
                     seed=args.seed)
    return {
        "task": override(TaskConfig(), train_fraction=args.train_fraction, seed=args.seed),
        "train": train,
        "embedding": EmbeddingConfig(),
        "interface": InterfaceConfig(),
        "interpretability": InterpretabilityConfig(),
        "experiment": override(ExperimentConfig(), device=args.device),
        "data": DataConfig(),
        "dynamics": DynamicsConfig(),
        "scale": ScaleConfig(),
        "wandb": override(WandbConfig(), mode=args.wandb_mode, project=args.wandb_project,
                          entity=args.wandb_entity),
    }


def print_row(row: dict) -> None:
    print(f"epoch {row['epoch']:3d}  train loss {row['train_loss']:.4f} acc {row['train_acc']:.3f}  "
          f"test loss {row['test_loss']:.4f} acc {row['test_acc']:.3f}  "
          f"grad {row['grad_norm']:.3g}  {row['seconds']:.1f} s  "
          f"gini emb {row['embedding_gini']:.3f}  drift {row['theta_drift']:.3g}", flush=True)


def print_and_log_row(run, row: dict) -> None:
    """on_row of run_experiment: print the row, then log it to W&B."""
    print_row(row)
    log_row(run, row)


def print_header(graph, model, configs: dict[str, object], run_dir) -> None:
    train, p = configs["train"], configs["embedding"].modulus
    n_train = round(configs["task"].train_fraction * p * p)
    n_channels, group_in = model.input_groups.shape
    print(f"N {graph.n_neurons}, E {graph.n_edges}, n_in {model.rnn.input_idx.numel()}, "
          f"channels {n_channels} x {group_in}, "
          f"n_out {model.rnn.output_idx.numel()}, T {configs['dynamics'].n_steps}, p {p}, "
          f"n_train {n_train}, batch {train.batch_size}, epochs {train.n_epochs}, "
          f"device {configs['experiment'].device}, run_dir {run_dir}", flush=True)


def print_summary(summary: dict) -> None:
    for name, table in summary.items():
        print(f"{name}: gini {table['gini']:.4f}, dominant frequencies {table['dominant']}")


def main() -> None:
    args = parse_args()
    configs = build_configs(args)
    data, ecfg = configs["data"], configs["experiment"]
    graph = load_graph(graph_cache_path(data.processed_dir, data.min_syn_count))
    weights = random_trainable_weights(graph, configs["scale"])
    model = build_group_interface_model(graph, weights, configs["dynamics"], configs["embedding"],
                                        configs["interface"], configs["train"].seed).to(ecfg.device)
    run_dir = ecfg.runs_dir / args.run_name
    print_header(graph, model, configs, run_dir)
    run = start_run(configs["wandb"], args.run_name, configs)
    run_experiment(model, configs, run_dir, on_row=partial(print_and_log_row, run))
    summary = spectra_summary(model, configs["interpretability"])
    print_summary(summary)
    finish_run(run, run_dir, summary, configs["wandb"], ecfg)


if __name__ == "__main__":
    main()
