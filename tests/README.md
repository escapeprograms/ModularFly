# tests

Unit tests on tiny random graphs (40 neurons, 300 connections). They need no FlyWire data and no GPU. Run from the repository root:

```
python -m pytest tests/ -q
```

## conftest.py (shared helpers)

| Signature | Returns | What it does |
|---|---|---|
| `random_graph(n_neurons, n_edges, seed)` | `ConnectomeGraph` | A random graph with no self-connections, random signs (about 30% inhibitory), and random labels at every annotation level, some missing. |
| `dense_matrix(graph, weights)` | `FloatTensor[N, N]` | The dense matrix `W[post, pre] = weights`, used as a reference. |
| `SMALL_DYN` | `DynamicsConfig` | Short dynamics for tests: 6 steps, readout over the last 2, 32-connection gradient chunks. |
| `make_rnn(graph, weights)` | `(ConnectomeRNN, n_in, n_out)` | A `ConnectomeRNN` with `SMALL_DYN` that drives the afferent neurons and reads the efferent ones. |
| `small_graph` (fixture) | `ConnectomeGraph` | `random_graph(40, 300, seed=0)`. |

## What each test checks

**test_graph.py**

| Test | Checks |
|---|---|
| `test_edges_sorted_by_post_then_pre` | Connections are stored sorted by `(post, pre)`. |
| `test_duplicate_edges_rejected` | A duplicate connection raises. |
| `test_in_degree` | `in_degree()` matches a direct count. |
| `test_save_load_roundtrip` | A saved and reloaded graph has the same connections, vocabulary and labels. |

**test_groups.py**

| Test | Checks |
|---|---|
| `test_select_neurons` | Selection returns exactly the neurons with the requested labels. |
| `test_group_neurons_falls_back_to_coarser_level` | Neurons without a label at a level are grouped by a coarser one. |
| `test_group_neurons_labeled_neurons_group_by_label` | Labeled neurons share a group exactly when they share a label. |
| `test_pair_gain_index_only_occupied_pairs` | Only group pairs that have a connection get a gain slot. |
| `test_baseline_indices` | The global and excitatory/inhibitory indices have 1 and 2 slots. |

**test_propagate.py**

| Test | Checks |
|---|---|
| `test_forward_matches_dense` | The sparse product equals the dense `W @ h`. |
| `test_gradcheck_float64` | Hand-written gradients pass PyTorch's numerical gradient check. |
| `test_gradients_match_dense_reference` | Gradients for weights and state equal those of the dense product. |
| `test_frozen_weights_get_no_gradient` | Frozen weights receive no gradient. |

**test_scaling.py**

| Test | Checks |
|---|---|
| `test_spectral_radius_matches_eigvals` | The estimate matches the true largest eigenvalue magnitude. |
| `test_spectral_radius_complex_dominant_pair` | It is still right when the largest eigenvalues are a complex pair. |

**test_model.py**

| Test | Checks |
|---|---|
| `test_condition2_equals_condition1_at_init` | Gains start at 1, so gain-modulated weights equal the frozen ones before training. |
| `test_condition3_keeps_presynaptic_sign_after_step` | Random weights keep each source neuron's sign after an optimizer step. |
| `test_forward_shape_and_gradients` | `forward` returns `[B, n_out]` and gradients flow. |
| `test_iter_states_yields_t_states` | `iter_states` yields `T` states of shape `[N, B]`. |
| `test_window_mean_matches_forward` | `window_mean` on the output neurons, transposed, equals `forward`; on other neurons it equals the mean of the last `readout_steps` states from `iter_states`. |

**test_embedding.py**

| Test | Checks |
|---|---|
| `test_input_embedding_shares_one_table` | Both numbers look up the same table: equal tokens give equal rows, and a row equals the table entry. |
| `test_input_embedding_init_std` | Entries start with the requested standard deviation. |
| `test_unembedding_shape_and_no_bias` | The output layer maps `[B, n_out]` to `[B, p]` and has no bias. |

**test_split_input.py**

| Test | Checks |
|---|---|
| `test_forward_shape` | `forward` maps tokens `[B, 2]` to logits `[B, p]`. |
| `test_halves_partition_input_slots` | The two halves are disjoint, cover all but one of the 15 inputs, leave that one undriven, and carry each number's embedding. |
| `test_same_token_gives_same_drive_on_both_halves` | Tokens `(k, k)` drive both halves identically. |
| `test_split_is_seeded` | The same seed gives the same split; a different seed gives a different one. |
| `test_gradients_reach_embedding_unembedding_and_theta` | Gradients reach both layers and the fly weights, `slots` is not trainable, and unused tokens get zero gradient. |
| `test_frozen_fly_trains_only_embeddings` | With frozen natural weights only the two layers train. |

**test_rewire.py**

| Test | Checks |
|---|---|
| `test_rewire_preserves_degrees_and_simplicity` | Rewiring keeps every in- and out-degree, creates no duplicate or self-connections, and changes the connection set. |
| `test_rewire_keeps_out_strength` | Each neuron's total outgoing synapse count is unchanged. |

**test_fourier.py** (uses `p = 13`, float64)

| Test | Checks |
|---|---|
| `test_basis_orthonormal` | `F @ F.T` is the identity for `p = 7` and `p = 113`. |
| `test_basis_rejects_even_p` | An even `p` raises `ValueError`. |
| `test_pair_sum` | Cos/sin pairs are added correctly, on 1D input and along a chosen dim. |
| `test_cosine_weight_has_single_frequency` | A weight `cos(w_2 x)·r` has a non-zero norm only at `k = 2`. |
| `test_frequency_component_reconstructs` | The components for `k = 0..K` add up to the original weight; `k = K+1` raises. |
| `test_sum_grid_is_sum_dominated` | A `cos(w(a+b))` grid puts all its energy in frequency `k`, all of it on the sum direction; a `cos(w(a−b))` grid puts its energy on the difference direction. |
| `test_sum_plus_diff_equals_block_energy` | Sum plus difference energy equals the `(k,k)` block energy for random grids. |
| `test_block_energy_total` | The block energies add up to the total squared coefficients. |
| `test_fourier_2d_batched_matches_loop` | The batched 2D transform equals one grid at a time. |

**test_embedding_spectrum.py**

| Test | Checks |
|---|---|
| `test_gini_one_hot_and_uniform` | Gini is `1 − 1/n` for one-hot, 0 for uniform and 0 for all zeros. |
| `test_dominant_frequencies` | Frequencies at or above half the largest non-constant norm are returned; the large constant is ignored. |
| `test_embedding_spectrum_synthetic` | A single-frequency weight gives that frequency as the only dominant one, zero norm elsewhere, and Gini `1 − 1/K`. |
| `test_model_token_spectra_shapes` | On a `SplitInputModel`, the embedding and unembedding spectra have the expected shapes and carry no gradient. |

**test_neurons.py** (uses `p = 7` on the small graph)

| Test | Checks |
|---|---|
| `test_all_pairs_order` | Pairs are row-major: row `a*p + b` is `(a, b)`. |
| `test_activation_grid_matches_iter_states` | Grid entries equal the readout-window mean computed directly from `iter_states`, with an uneven last chunk. |
| `test_readout_grid_matches_forward` | The readout grid of the output neurons equals `rnn.forward` on all pairs. |
| `test_chunking_is_invariant` | The grid is the same for different `pair_chunk` values. |
| `test_state_at_step` | Step 0 equals the first state from `iter_states`; steps outside `0..T-1` raise. |
| `test_grid_has_no_grad` | The collected grid carries no gradient. |
| `test_neuron_spectra_synthetic` | `cos(w(a+b))`, `cos(w(a−b))` and constant grids give the right dominant frequency, `fve` of 1, 1, 0 and `sum_share` of 1, 0, 0, with no NaN. |
| `test_operand_only_neuron_has_zero_sum_share` | A grid that depends on `a` only has `fve` 1 and `sum_share` exactly 0. |
| `test_cluster_by_frequency` | Only neurons above the FVE threshold are clustered, under their dominant frequency. |

**test_plots.py**

The plot functions use matplotlib's object-oriented API (`Figure`, not pyplot), so these tests need no display or backend.

| Test | Checks |
|---|---|
| `test_frequency_norms_has_one_bar_per_frequency` | One axes with `K+1` bars. |
| `test_input_profile_has_one_bar_per_input` | One axes with `p` bars. |
| `test_activation_grid_heatmap_and_colorbar` | A heatmap axes plus a colorbar axes; the image is `[p, p]`. |
| `test_fourier_grid_heatmap_and_colorbar` | A heatmap plus a colorbar; the image is `[p, p]` and non-negative (absolute values). |
| `test_fve_histogram_uses_configured_bins` | The histogram has `cfg.histogram_bins` bars. |
| `test_save_figure_writes_png` | `save_figure` creates missing directories, writes a non-empty file and returns its path. |
