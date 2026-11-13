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
