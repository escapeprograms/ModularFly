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
    modulus: int = 113  # p: tokens 0..p-1 and p answers (Nanda et al. 2023 setup)
    init_std: float = 1.0  # embedding entries ~ N(0, std^2); each entry is one neuron's drive


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
