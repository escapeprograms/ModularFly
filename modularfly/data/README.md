# modularfly/data

Turns the two FlyWire files into a `ConnectomeGraph`, and provides the operations on that graph: choosing neurons, grouping them, and rewiring.

| File | Job |
|---|---|
| [graph.py](graph.py) | The `ConnectomeGraph` container, and saving/loading it. |
| [download.py](download.py) | Network access only: fetch the raw files and check them. |
| [flywire.py](flywire.py) | File reading only: raw files → `ConnectomeGraph`. |
| [groups.py](groups.py) | Select neurons, group neurons, map connections to shared gains. |
| [rewire.py](rewire.py) | Degree-preserving rewired control graph. |

`N` = neurons, `E` = connections. Connection `e` carries activity from `pre[e]` to `post[e]`.

## graph.py

### `ConnectomeGraph` (dataclass)

| Field | Shape / type | Meaning |
|---|---|---|
| `n_neurons` | `int` | `N`. |
| `pre`, `post` | `LongTensor[E]` | Source and target neuron index of each connection. |
| `syn_count` | `FloatTensor[E]` | Synapses in each connection. |
| `neuron_sign` | `FloatTensor[N]` | `+1` excitatory, `-1` inhibitory, from the predicted transmitter. |
| `nt_confidence` | `FloatTensor[N]` | Confidence of that prediction; `0` where there is none. |
| `annotations` | `dict[str, LongTensor[N]]` | Integer label code per neuron for each level (`flow`, `super_class`, `cell_class`, `cell_sub_class`, `supertype`, `cell_type`, `side`, `top_nt`). `-1` means no label. |
| `vocab` | `dict[str, list[str]]` | Code → label text, per level. |
| `root_ids` | `LongTensor[N]` | FlyWire root id of each neuron. |

Constructing a graph always sorts connections by `(post, pre)`. It raises `ValueError` on duplicate connections, wrong shapes, or out-of-range indices.

| Member | Returns | What it does |
|---|---|---|
| `graph.n_edges` | `int` | Number of connections. |
| `graph.in_degree()` | `LongTensor[N]` | Incoming connections per neuron. |
| `graph.out_degree()` | `LongTensor[N]` | Outgoing connections per neuron. |
| `graph.with_edges(pre, post, syn_count)` | `ConnectomeGraph` | Same neurons with a new connection list (re-sorted). |

### Functions

| Signature | Returns | What it does |
|---|---|---|
| `edge_keys(major, minor, n_neurons)` | `LongTensor` | One unique integer per `(major, minor)` pair, used for sorting and duplicate checks. |
| `graph_to_dict(graph)` | `dict` | The graph's fields as a plain dict. |
| `graph_cache_path(processed_dir, min_syn_count)` | `Path` | `<processed_dir>/flywire783_graph_min<min_syn_count>.pt`. |
| `save_graph(graph, path)` | `None` | Saves the graph as a dict of tensors and lists. |
| `load_graph(path)` | `ConnectomeGraph` | Loads a saved graph. |

## download.py

| Signature | Returns | What it does |
|---|---|---|
| `file_md5(path, chunk_bytes)` | `str` | md5 of a file. |
| `check_md5(path, expected, chunk_bytes)` | `None` | Raises `ValueError` if the md5 differs. |
| `fetch(url, dest, chunk_bytes)` | `None` | Downloads to `dest` (through a `.part` file). |
| `ensure_file(remote, raw_dir, chunk_bytes)` | `Path` | Downloads the file if absent, then checks its md5 either way. |
| `ensure_flywire_files(cfg)` | `(Path, Path)` | Paths of the connections and annotation files, downloaded if needed. |

## flywire.py

| Signature | Returns | What it does |
|---|---|---|
| `read_connections(path)` | `DataFrame` | Reads the pre id, post id and synapse count columns; raises if ids are not int64. |
| `read_annotations(path, cfg)` | `DataFrame` | Reads root id, label columns and transmitter confidence; raises on duplicate root ids. |
| `aggregate_connections(connections)` | `DataFrame` | Sums synapse counts over brain regions: one row per (pre, post). |
| `threshold_connections(connections, min_syn_count)` | `DataFrame` | Keeps rows with at least `min_syn_count` synapses. |
| `drop_unannotated_edges(connections, root_ids)` | `(DataFrame, int)` | Removes connections whose pre or post neuron is not in the annotation table; the int is how many were removed (84 on the current files). |
| `index_edges(connections, root_ids)` | `(LongTensor[E], LongTensor[E])` | Converts root ids to neuron indices; raises if any endpoint is unknown. |
| `encode_labels(labels)` | `(LongTensor[N], list[str])` | Label text → integer codes plus the sorted vocabulary; missing → `-1`. |
| `neuron_signs(nt_labels, cfg)` | `FloatTensor[N]` | Transmitter → `±1`; raises on a transmitter with no sign defined. |
| `encode_annotations(annotations, cfg)` | `(dict, dict)` | Codes and vocabularies for every label column. |
| `load_edges(connections_path, root_ids, cfg)` | `(pre, post, syn_count, n_dropped)` | Read → aggregate → threshold → drop unannotated → index. |
| `build_graph(connections_path, annotations_path, cfg)` | `(ConnectomeGraph, int)` | The full build; the int is the number of dropped connections. |

## groups.py

| Signature | Returns | What it does |
|---|---|---|
| `select_neurons(graph, level, labels)` | `LongTensor` | Sorted indices of neurons whose label at `level` is in `labels`, e.g. `("flow", ["afferent"])`. Raises `KeyError` for an unknown label. |
| `group_neurons(graph, level, hierarchy=...)` | `LongTensor[N]` | A group id per neuron at `level`. A neuron with no label there uses its label at the next coarser level instead. |
| `pair_gain_index(graph, groups)` | `(edge_group[E], pair_keys[P, 2])` | One gain slot per (post group, pre group) pair that actually has a connection. `edge_group` gives each connection's slot; `pair_keys` gives each slot's (post group, pre group). |
| `global_gain_index(graph)` | `(edge_group[E], keys[1])` | Every connection shares one gain slot. |
| `sign_gain_index(graph)` | `(edge_group[E], keys[2])` | Two slots: connections from inhibitory neurons (`0`) and from excitatory neurons (`1`). |

Each `*_gain_index` result feeds `GroupGain(edge_group, len(keys))` in [model/weights/gain_modulated.py](../model/weights/gain_modulated.py).

## rewire.py

Builds a control graph by repeatedly swapping the targets of two random connections: `(a→b), (c→d)` becomes `(a→d), (c→b)`. Every neuron keeps its number of incoming and outgoing connections. Each connection keeps its source neuron and synapse count, so every neuron's sign and total outgoing strength are unchanged; total incoming strength is not.

| Signature | Returns | What it does |
|---|---|---|
| `rewire(graph, cfg, device=None)` | `RewireResult` | Runs rounds until `swaps_per_edge × E` swaps are accepted. Raises `RuntimeError` after `max_rounds`. |
| `RewireResult` | dataclass | `graph` (rewired), `rounds`, `proposed`, `accepted`. |
| `rewire_round(pre, post, n_neurons, remaining, generator)` | `int` | One round; edits `post` in place and returns the swaps applied. |
| `propose_pairs(n_edges, generator, device)` | `(LongTensor, LongTensor)` | Random disjoint pairs of connection slots. |
| `accepted_swaps(pre, post, i, j, n_neurons)` | `BoolTensor` | Which proposed swaps are allowed. |
| `endpoint_mask(a, b, c, d)` | `BoolTensor` | Rejects swaps that involve or create a self-connection, or change nothing. |
| `contains(sorted_keys, query)` | `BoolTensor` | Whether each query key already exists. |
| `collision_free(keys_ad, keys_cb, candidate)` | `BoolTensor` | Rejects swaps that would create the same connection twice in one round. |

On the real graph, 5 swaps per connection takes about 1 s on GPU or 11 s on CPU. After rewiring, rebuild any gain index from the new graph.
