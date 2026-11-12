# Design debt
Only for debt spanning several files. Single-spot debt stays as `# TODO(debt-NNN): ...` in code.
Delete an entry when it's fixed. Here is an example

## debt-001: Sample rate hard-coded in loader and model
- Where: `data/loaders.py::MelLoader`, `models/flow.py::FlowModel.__init__`
- Why: quick fix during task 3 to get training running
- Fix when: before supporting a second sample rate or dataset
- Fix: read from `config.audio.sample_rate`, pass in at construction
- Done when: grep for `22050` finds no results outside config


# list of technical debt
## debt-001: Scripts put the repo root on sys.path by hand
- Where: `scripts/download_flywire.py`, `scripts/build_graph.py`, `scripts/smoke_gpu.py`
- Why: task 1 has no packaging metadata, so `modularfly` is not installed in the env
- Fix when: a pyproject.toml is added, or a fourth script is written
- Fix: add a minimal pyproject.toml, `pip install -e .` in `modular-fly`, delete the sys.path lines
- Done when: grep for `sys.path.insert` finds no results
