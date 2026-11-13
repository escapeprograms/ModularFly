"""All tunables for ModularFly. Later tasks change values here, not inline."""
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RemoteFile:
    url: str
    relative_path: str  # under DataConfig.raw_dir, one directory per source
    md5: str


CONNECTIONS_783 = RemoteFile(
    url="https://zenodo.org/records/10676866/files/proofread_connections_783.feather?download=1",
    relative_path="zenodo_10676866/proofread_connections_783.feather",
    md5="f48f972d262323a102aed49af1396b8a",
)

# This file is served from the repo's `main` branch, so upstream edits change its md5.
# A mismatch makes download.py raise; re-check the data and update the hash deliberately.
NEURON_ANNOTATIONS = RemoteFile(
    url="https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/"
    "supplemental_files/Supplemental_file1_neuron_annotations.tsv",
    relative_path="flywire_annotations/Supplemental_file1_neuron_annotations.tsv",
    md5="fd4fcb205256ca2f80a2eca3a0cc5d87",
)

# Shiu 2024: ACh excitatory; GABA and glutamate inhibitory; monoamines excitatory.
NT_SIGN = {
    "acetylcholine": 1.0,
    "glutamate": -1.0,
    "gaba": -1.0,
    "dopamine": 1.0,
    "serotonin": 1.0,
    "octopamine": 1.0,
}


@dataclass
class DataConfig:
    raw_dir: Path = REPO_ROOT / "data" / "raw"
    processed_dir: Path = REPO_ROOT / "data" / "processed"
    connections: RemoteFile = CONNECTIONS_783
    annotations: RemoteFile = NEURON_ANNOTATIONS
    download_chunk_bytes: int = 1 << 20
    min_syn_count: int = 5  # Dorkenwald 2024 connection threshold
    # Annotation hierarchy, coarsest first (Schlegel 2024). group_neurons falls back along it.
    annotation_levels: tuple[str, ...] = (
        "flow", "super_class", "cell_class", "cell_sub_class", "supertype", "cell_type",
    )
    extra_annotation_columns: tuple[str, ...] = ("side", "top_nt")
    nt_column: str = "top_nt"
    nt_confidence_column: str = "top_nt_conf"
    nt_sign: dict[str, float] = field(default_factory=lambda: dict(NT_SIGN))
    # 602 neurons in the v783 annotations have no top_nt; they are treated as excitatory
    # (the majority class, acetylcholine) and get nt_confidence 0.
    missing_nt_sign: float = 1.0
    # select_neurons I/O filters: used by build_graph.py, smoke_gpu.py and tests; experiment 1
    # samples its interface from any connected neuron instead (InterfaceConfig).
    input_level: str = "flow"
    input_labels: tuple[str, ...] = ("afferent",)  # sensory + ascending
    output_level: str = "flow"
    output_labels: tuple[str, ...] = ("efferent",)  # motor + endocrine + descending


@dataclass
class DynamicsConfig:
    alpha: float = 0.5  # leak: h <- (1 - alpha) h + alpha tanh(u + W h + b)
    n_steps: int = 64
    readout_steps: int = 4  # readout = mean output activity over the last K steps
    propagate_chunk_edges: int = 1 << 18  # edges per chunk in the per-edge weight gradient


@dataclass
class ScaleConfig:
    target_spectral_radius: float = 1.0  # w0 = target / rho(sign * syn_count)
    power_iterations: int = 200
    inhibitory_init_scale: float = 1.0  # 1.0 is paper-faithful (Cho 2026)
    seed: int = 0


@dataclass
class RewireConfig:
    swaps_per_edge: int = 5  # Cho 2026 App. D
    max_rounds: int = 200
    seed: int = 0


@dataclass
class EmbeddingConfig:
    modulus: int = 59 #McCraken
    channels_per_token: int = 22  # embedding width per operand; 2 operands give 44 channels (arXiv 2610.10014)
    init_std: float = 1.0  # embedding entries ~ N(0, std^2); each entry is one channel's drive


@dataclass
class InterfaceConfig:
    input_group_size: int = 64  # neurons driven by each input channel (paper default)
    output_group_size: int = 32  # neurons averaged into each class logit (paper default)
    input_amplitude: float = 1.0  # each channel is copied onto its group times this (paper default)
    temperature: float = 0.1  # logit = group-mean readout / temperature (paper default)
    seed: int = 0  # interface sampling seed, independent of TrainConfig.seed (paper: fixed across seeds)


@dataclass
class InterpretabilityConfig:
    pair_chunk: int = 256  # (a, b) pairs per forward pass: 256 x N x 4 B ~ 143 MB per state on the real fly
    dominant_fraction: float = 0.5  # a frequency is dominant if its norm >= this * the largest non-constant norm
    fve_threshold: float = 0.85  # single-frequency neuron if its best k explains this fraction (Nanda 2023)
    histogram_bins: int = 20
    figure_size: tuple[float, float] = (6.0, 4.0)
    heatmap_size: tuple[float, float] = (5.0, 4.5)
    dpi: int = 150
    figure_dir: Path = REPO_ROOT / "figures"


@dataclass
class TaskConfig:
    train_fraction: float = 0.5   # 0.3 is Nanda 2023's grokking regime; 0.5 first, to check the connectome can learn and generalize at all
    seed: int = 0                 # the split permutation


@dataclass
class TrainConfig:
    batch_size: int = 32          # 64 [N, B] states kept for backprop; 2.41 GB peak on the real fly
    eval_batch: int = 512         # no_grad: ~1 GB peak at B=512
    lr: float = 1e-3
    fly_lr: float = 1e-3          # learning rate of the recurrent weights (model.rnn); lr is the token embedding's. Each theta moves at most fly_lr per step, so 3e-3..1e-2 is worth trying
    betas: tuple[float, float] = (0.9, 0.98)
    weight_decay: float = 0.1     # token embedding; AdamW shrinks it by lr*wd per step, and 1.0 (Nanda 2023) decays a [59, 22] table ~5x over a 50-epoch run
    fly_weight_decay: float = 0.0  # recurrent weights; 1.0 would shrink every theta by 0.999/step and silence the dynamics
    fly_penalty: float = 0.0      # lambda * sum((theta - theta_init)^2); off by default. Measured: 1e-3 pins theta (drift plateau ~90 = RMS 0.006 per edge); the plateau scales ~1/lambda, so sweep 1e-6..1e-5
    grad_clip_norm: float = 1.0   # arXiv 2610.10014 clips at 1; the pre-clip norm is always logged
    n_epochs: int = 50            # ~1 h on the real fly at p=59 (55 steps/epoch at train_fraction 0.5, 78 s/epoch measured); raise from the CLI
    seed: int = 0                 # model init (torch.manual_seed) and batch order


@dataclass
class ExperimentConfig:
    runs_dir: Path = REPO_ROOT / "runs"
    device: str = "cuda"
    config_name: str = "config.json"
    history_name: str = "history.csv"
    checkpoint_name: str = "checkpoint.pt"
    report_name: str = "embedding_report.json"
    figure_dir_name: str = "figures"


@dataclass
class WandbConfig:
    entity: str = "archimedesli"
    project: str = "ModularFly"
    mode: str = "online"            # passed straight to wandb: online | offline | disabled
    env_file: Path = REPO_ROOT / ".env"
    key_variable: str = "WANDB_KEY"
    artifact_name: str = "run-files"  # config.json, history.csv, embedding_report.json, figures/
