import json

from matplotlib.figure import Figure

from modularfly.config import ExperimentConfig, InterpretabilityConfig
from modularfly.experiments.report import history_figures, spectra_summary, write_report
from tests.conftest import make_model, trainable_fly

ROWS = [
    {"epoch": e, "train_loss": 2.0 - e, "train_acc": 0.1 * e, "test_loss": 2.1 - e,
     "test_acc": 0.05 * e, "grad_norm": 1.0, "theta_drift": 0.5 * e, "seconds": 0.1,
     "embedding_gini": 0.2}
    for e in range(3)
]


def test_spectra_summary_is_json_able(small_graph):
    summary = spectra_summary(make_model(small_graph, trainable_fly(small_graph)),
                              InterpretabilityConfig())
    assert set(summary) == {"embedding"}
    for table in summary.values():
        assert set(table) == {"gini", "dominant", "freq_norms"}
        assert len(table["freq_norms"]) == 7 // 2 + 1
    json.dumps(summary)


def test_history_figures_keys_and_lines():
    figures = history_figures(ROWS, InterpretabilityConfig())
    assert set(figures) == {"loss", "accuracy", "gini", "grad_norm", "theta_drift"}
    assert all(isinstance(fig, Figure) for fig in figures.values())
    n_lines = {name: len(fig.axes[0].get_lines()) for name, fig in figures.items()}
    assert n_lines == {"loss": 2, "accuracy": 2, "gini": 1, "grad_norm": 1, "theta_drift": 1}
    assert list(figures["loss"].axes[0].get_lines()[0].get_xdata()) == [0, 1, 2]


def test_write_report_writes_json_and_pngs(small_graph, tmp_path):
    ecfg = ExperimentConfig(runs_dir=tmp_path)
    model = make_model(small_graph, trainable_fly(small_graph))
    write_report(tmp_path, model, ROWS, InterpretabilityConfig(), ecfg)
    report = json.loads((tmp_path / ecfg.report_name).read_text())
    assert set(report) == {"embedding"}
    pngs = sorted(p.name for p in (tmp_path / ecfg.figure_dir_name).glob("*.png"))
    assert pngs == sorted(["loss.png", "accuracy.png", "gini.png", "grad_norm.png",
                           "theta_drift.png", "embedding_freq_norms.png"])
