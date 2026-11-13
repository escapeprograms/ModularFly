# tests

Unit tests on tiny random graphs (40 neurons, 300 connections). They need no FlyWire data and no GPU. The suite has 100 tests. Run from the repository root:

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
| `SMALL_P` | `int` | Modulus `p = 7` for the `GroupInterfaceModel` tests. |
| `SMALL_EMB` | `EmbeddingConfig` | `modulus=SMALL_P`, `channels_per_token=2`: 4 input channels. |
| `SMALL_INTERFACE` | `InterfaceConfig` | `input_group_size=3`, `output_group_size=2`: 12 input and 14 output neurons of the 40 (every small-graph neuron has both in- and out-degree > 0). |
| `make_model(graph, weights, seed=0)` | `GroupInterfaceModel` | `build_group_interface_model` with `SMALL_DYN`, `SMALL_EMB` and `SMALL_INTERFACE`; `seed` seeds both the embedding init and the interface sampling. |
| `trainable_fly(graph)` | `SignedEdgeWeights` | Random trainable signed weights (inhibitory scale 1, seed 0). |
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
| `test_drift_penalty_zero_at_init_and_positive_after_change` | `SignedEdgeWeights.drift_penalty()` is 0 at init; after `theta += 0.1` it is a 0-dim tensor with grad equal to `0.01 · E`; `theta_init` is in `state_dict()` and still equals the initial `theta`. |
| `test_frozen_and_gain_drift_penalties` | `SynapseCountWeights` gives 0 with no grad; `GroupGain` gives `sum(z²)`; `GainModulatedWeights` gives the sum of its base and gain penalties. |

**test_embedding.py**

| Test | Checks |
|---|---|
| `test_input_embedding_shares_one_table` | Both numbers look up the same table: equal tokens give equal rows, and a row equals the table entry. |
| `test_input_embedding_init_std` | Entries start with the requested standard deviation. |

**test_interface.py** (4 channels × 3 inputs, 7 classes × 2 outputs on the small graph)

| Test | Checks |
|---|---|
| `test_interface_shapes` | `sample_interface` returns input groups `(4, 3)` and output groups `(7, 2)`. |
| `test_groups_are_disjoint` | 12 distinct inputs, 14 distinct outputs, and no neuron in both. |
| `test_candidates` | `input_candidates` equals `graph.pre.unique()`; `output_candidates` is `graph.post.unique()` minus the excluded ids, in order. |
| `test_sampling_is_seeded` | The same generator seed gives the same groups; a different seed gives different ones. |
| `test_too_few_candidates_raise` | Input groups of 20 (80 of 40) and output groups of 5 (35 of the 28 left after the inputs) raise `ValueError`. |
| `test_sample_groups_without_replacement` | `sample_groups` returns the requested shape with distinct ids, all from the candidates. |
| `test_group_drive_repeats_each_channel` | Each channel value is repeated `group_size` times and multiplied by the amplitude. |
| `test_group_mean_logits_pairwise_means` | Logits are the means of consecutive `group_size` blocks divided by the temperature. |

**test_group_interface.py** (`make_model`: 4 channels, groups of 3 inputs and 2 outputs, `p = 7`)

| Test | Checks |
|---|---|
| `test_forward_shape` | `forward` maps tokens `[B, 2]` to logits `[B, p]`. |
| `test_drive_copies_each_channel_onto_its_group` | The drive is constant within each input group and equals `embedding(tokens).flatten(1)` per channel. |
| `test_same_token_gives_same_drive_on_both_operands` | Tokens `(k, k)` give equal drive on both operands' halves. |
| `test_logits_are_group_means_over_temperature` | `model(tokens)` equals the RNN readout averaged over each output group, divided by 0.1. |
| `test_groups_view_rnn_buffers` | `input_groups` `(4, 3)` and `output_groups` `(7, 2)` flatten to `rnn.input_idx` and `rnn.output_idx`. |
| `test_gradients_reach_embedding_and_theta` | Gradients reach `embedding.weight` and `theta`, unused tokens get zero gradient, and the trainable parameters are exactly `embedding.weight` and `rnn.edge_weights.theta`. |
| `test_frozen_fly_trains_only_embedding` | With frozen natural weights only `embedding.weight` trains. |
| `test_mismatched_rnn_raises` | Wrapping an RNN with the wrong number of input neurons (the afferent `make_rnn`) raises `ValueError`. |

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
| `test_model_embedding_spectrum_shapes` | On a `GroupInterfaceModel`, `model_embedding_spectrum` gives `coeffs [p, 2]` and `freq_norms [K+1]`, with no gradient. |

**test_neurons.py** (uses `p = 7` on the small graph, through conftest's `make_model`)

| Test | Checks |
|---|---|
| `test_all_pairs_order` | Pairs are row-major: row `a*p + b` is `(a, b)`. |
| `test_activation_grid_matches_iter_states` | Grid entries equal the readout-window mean computed directly from `iter_states`, with an uneven last chunk. |
| `test_readout_grid_matches_forward` | The readout grid of the 14 output neurons equals `rnn.forward` of `model.drive` on all pairs. |
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

**test_task.py** (uses `p = 7`)

| Test | Checks |
|---|---|
| `test_split_sizes_disjoint_and_complete` | A 0.3 split of the 49 pairs has 15 training and 34 test rows, with no overlap and together all of `all_pairs(7)`. |
| `test_targets_are_sum_mod_p` | Every target is `(a + b) % 7`, in `labeled_pairs` and in both halves of the split. |
| `test_split_is_seeded` | The same seed gives the same split; a different seed gives a different one. |
| `test_to_returns_new_split` | `SplitData.to` returns a new object with the same tensors. |

**test_loop.py** (uses `p = 7` on the small graph with random trainable weights)

| Test | Checks |
|---|---|
| `test_fresh_model_logits_bounded_by_inverse_temperature` | An untrained model's logits stay below `1 / temperature` in magnitude (tanh states lie in `(−1, 1)`), its test loss is finite and its accuracy is in `[0, 1]`. |
| `test_train_steps_lower_fixed_batch_loss` | Five `train_step`s (`penalty_weight=0`) at learning rate 0.1 (`lr` and `fly_lr`) on one fixed batch lower its loss. |
| `test_iter_epochs_yields_records` | Three epochs yield records with `epoch` 0, 1, 2, finite losses and grad norm, and positive `seconds`. |
| `test_same_seed_runs_give_equal_records` | Two runs with the same model init and seeds give equal records (apart from `seconds`, `theta_drift` included) on the CPU. |
| `test_penalty_pulls_theta_toward_init` | With `theta` pushed 0.5 off `theta_init`, one step at learning rate 0.1 (`lr` and `fly_lr`) on a fixed batch leaves a smaller drift penalty at `penalty_weight=1e2` than at 0. |
| `test_records_log_theta_drift` | With trainable weights `theta_drift` is positive and finite after epoch 0; with frozen natural weights every record has `theta_drift == 0.0`. |
| `test_optimizer_groups` | `theta` is alone in the fly group at weight decay 0; `embedding.weight` is alone in the token group at weight decay 0.1. With the default config both groups have lr 1e-3. |
| `test_fly_lr_sets_fly_group_only` | `TrainConfig(lr=1e-3, fly_lr=5e-3)` gives the token group lr 1e-3 and the fly group lr 5e-3. |
| `test_frozen_fly_has_only_token_group` | With frozen natural weights the optimizer has one group holding only the embedding. |
| `test_evaluate_uneven_chunk_matches_single_chunk` | Evaluating 34 rows in chunks of 16 gives the same loss and accuracy as one chunk. |

**test_checkpoint.py** (uses `p = 7`)

| Test | Checks |
|---|---|
| `test_roundtrip_restores_model_and_epoch` | Models are built with `build_group_interface_model` and seeded `SignedEdgeWeights`. After one `train_step` at the default `fly_penalty`, a saved checkpoint loads with `weights_only=True`; a model built with another seed gets equal state-dict tensors (including `theta_init`) and equal logits, the epoch round-trips, and the AdamW state loads into a fresh optimizer. |

**test_fly_model.py** (uses `p = 7` on the small graph)

| Test | Checks |
|---|---|
| `test_random_weights_train_fly_and_tokens` | With `random_trainable_weights` the trainable parameters are `embedding.weight` and `rnn.edge_weights.theta`. |
| `test_frozen_weights_train_only_tokens` | With `SynapseCountWeights` only `embedding.weight` trains. |
| `test_same_seed_gives_equal_embedding` | The same seed gives equal embedding weights; another seed gives different ones. |
| `test_interface_fixed_across_training_seeds` | Training seeds 3 and 4 give the same `input_idx`/`output_idx` but different embeddings, and `input_idx` equals `sample_interface` with `SMALL_INTERFACE.seed`. |

**test_report.py** (uses `p = 7` and three synthetic history rows)

| Test | Checks |
|---|---|
| `test_spectra_summary_is_json_able` | The summary has only `embedding`, with `gini`, `dominant` and `K+1` `freq_norms`, and passes `json.dumps`. |
| `test_history_figures_keys_and_lines` | Five figures `loss`, `accuracy`, `gini`, `grad_norm`, `theta_drift` with 2, 2, 1, 1 and 1 lines, plotted over the epochs. |
| `test_write_report_writes_json_and_pngs` | `write_report` writes the JSON report (only `embedding`) and the six named PNGs (including `theta_drift.png`). |

**test_run_experiment.py** (2 epochs, batch 8, train fraction 0.5, CPU, into `tmp_path`)

| Test | Checks |
|---|---|
| `test_config_json_has_every_config` | `config.json` parses and holds the five configs with their values. |
| `test_history_has_one_row_per_epoch_with_gini` | `history.csv` has rows for epochs 0 and 1 with every `GINI_COLUMNS` column and `theta_drift`; `on_row` got the same two rows that were returned. |
| `test_checkpoint_and_report_written` | `checkpoint.pt` loads with `load_checkpoint` at `epoch == 1`; the JSON report and six PNGs exist. |
| `test_existing_run_dir_raises` | An existing run directory raises `FileExistsError`. |

**test_wandb_logging.py** (W&B `mode="disabled"`, a dummy key in a temporary `.env`, never the real one)

| Test | Checks |
|---|---|
| `test_read_env_value` | The value of a variable in a dotenv file is returned. |
| `test_read_env_value_missing_name_raises` | A missing variable raises `KeyError`. |
| `test_read_env_value_empty_raises` | An empty value raises `KeyError`. |
| `test_disabled_run_logs_without_network` | `start_run`, `log_row` for both epochs of a 2-epoch `run_experiment`, and `finish_run` on its run directory complete; any non-loopback `socket.connect` fails the test (loopback is allowed for asyncio's self-pipe on Windows). |
