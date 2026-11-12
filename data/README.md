# data

Downloaded and derived data. This folder is git-ignored and can be rebuilt with `scripts/download_flywire.py` and `scripts/build_graph.py`. It holds no code.

## raw/

| File | Size | Source | Contents |
|---|---|---|---|
| `zenodo_10676866/proofread_connections_783.feather` | 852 MB | [Zenodo 10676866](https://zenodo.org/records/10676866) | FlyWire v783 connections. One row per (pre neuron, post neuron, brain region): `pre_pt_root_id`, `post_pt_root_id`, `neuropil`, `syn_count`, and six average transmitter scores. |
| `flywire_annotations/Supplemental_file1_neuron_annotations.tsv` | 32 MB | [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations) | One row per neuron: `root_id`, `flow`, `super_class`, `cell_class`, `cell_sub_class`, `supertype`, `cell_type`, `top_nt`, `top_nt_conf`, `side`, and more. |

The annotation file follows its repository's main branch. Its md5 is pinned in `modularfly/config.py`; if upstream changes it, the download script raises until the hash is updated.

## processed/

| File | Size | Contents |
|---|---|---|
| `flywire783_graph_min5.pt` | 65 MB | The cached `ConnectomeGraph` at the ≥5-synapse threshold: 139,248 neurons and 2,700,429 connections. Load it with `modularfly.data.graph.load_graph`. |

The number in the filename is the synapse threshold, so graphs built with different thresholds sit side by side.

## Known gaps

- The annotation table has 139,248 neurons, 7 fewer than the published 139,255.
- 84 connections are dropped because one endpoint is not in the annotation table.
- 602 neurons have no transmitter prediction; they are treated as excitatory with confidence 0.
