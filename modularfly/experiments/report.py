"""Summary and figures of a trained modular-addition model: the token-embedding spectrum
and the per-epoch training history.

The builders are pure (they return dicts and matplotlib Figures, object-oriented API, no
pyplot); write_report does every file write. `rows` are history rows: EpochRecord fields
plus GINI_COLUMNS, one dict per epoch.
"""
import json
from pathlib import Path

from matplotlib.figure import Figure

from modularfly.config import ExperimentConfig, InterpretabilityConfig
from modularfly.interpretability.embedding import EmbeddingSpectrum, model_embedding_spectrum
from modularfly.interpretability.plots import plot_frequency_norms, save_figure
from modularfly.model.end_to_end.group_interface import GroupInterfaceModel

GINI_COLUMNS = ("embedding_gini",)
TABLE_NAMES = ("embedding",)


def gini_row(model: GroupInterfaceModel, cfg: InterpretabilityConfig) -> dict[str, float]:
    """{"embedding_gini": ...} of the current weights."""
    return {"embedding_gini": model_embedding_spectrum(model, cfg).gini}


def _spectrum_summary(spectrum: EmbeddingSpectrum) -> dict:
    return {"gini": spectrum.gini, "dominant": list(spectrum.dominant),
            "freq_norms": spectrum.freq_norms.cpu().tolist()}


def spectra_summary(model: GroupInterfaceModel, cfg: InterpretabilityConfig) -> dict:
    """{"embedding": {"gini", "dominant", "freq_norms"}}, JSON-able."""
    return {"embedding": _spectrum_summary(model_embedding_spectrum(model, cfg))}


def plot_history(epochs: list[int], series: dict[str, list[float]], ylabel: str,
                 cfg: InterpretabilityConfig) -> Figure:
    """One line per series over the epochs, with a legend."""
    fig = Figure(figsize=cfg.figure_size)
    ax = fig.add_subplot()
    for label, values in series.items():
        ax.plot(epochs, values, label=label)
    ax.set(xlabel="epoch", ylabel=ylabel)
    ax.legend()
    return fig


def _columns(rows: list[dict], names: tuple[str, ...]) -> dict[str, list[float]]:
    return {name: [row[name] for row in rows] for name in names}


def history_figures(rows: list[dict], cfg: InterpretabilityConfig) -> dict[str, Figure]:
    """{"loss", "accuracy", "gini", "grad_norm", "theta_drift"} curves over the epochs."""
    epochs = [row["epoch"] for row in rows]
    panels = {
        "loss": (("train_loss", "test_loss"), "cross-entropy"),
        "accuracy": (("train_acc", "test_acc"), "accuracy"),
        "gini": (GINI_COLUMNS, "Gini of Fourier norms"),
        "grad_norm": (("grad_norm",), "mean pre-clip gradient norm"),
        "theta_drift": (("theta_drift",), "sum (theta - theta_init)^2"),
    }
    return {name: plot_history(epochs, _columns(rows, cols), ylabel, cfg)
            for name, (cols, ylabel) in panels.items()}


def spectra_figures(model: GroupInterfaceModel, cfg: InterpretabilityConfig) -> dict[str, Figure]:
    """{"embedding_freq_norms"}: per-frequency Fourier norm bars of the token embedding."""
    spectrum = model_embedding_spectrum(model, cfg)
    return {"embedding_freq_norms": plot_frequency_norms(spectrum.freq_norms, cfg)}


def write_report(run_dir: Path, model: GroupInterfaceModel, rows: list[dict],
                 icfg: InterpretabilityConfig, ecfg: ExperimentConfig) -> None:
    """Writes run_dir / ecfg.report_name (spectra_summary as JSON) and every history and
    spectra figure as run_dir / ecfg.figure_dir_name / <name>.png."""
    summary = spectra_summary(model, icfg)
    (run_dir / ecfg.report_name).write_text(json.dumps(summary, indent=2))
    figures = history_figures(rows, icfg) | spectra_figures(model, icfg)
    for name, fig in figures.items():
        save_figure(fig, run_dir / ecfg.figure_dir_name / f"{name}.png", icfg)
