# Training datasets (hosted)

These two CSVs are the **inputs every analysis reads**. They are the joined output of the
MQT_Loader pipeline (`src/mqtloader/run_pipeline.py`) over the in-distribution corpus, labeled
on `FakeBrisbane` for `N = 3-15`. The rest of `data/` is gitignored and regenerable; these two
are hosted because the fidelity arm is an expensive noisy simulation and is not bit-reproducible,
so a fresh clone should not have to rerun the HPC job to use the harness.

| file | arm | label column | rows | N | device |
|------|-----|--------------|------|---|--------|
| `train_swap_FakeBrisbane.csv` | swap (routing) | `bare_routed_2q` | 2827 | 3-15 | FakeBrisbane |
| `train_pol_FakeBrisbane.csv`  | fidelity        | `polarization`   | 2797 | 3-15 | FakeBrisbane |

## How they were produced (and how to reproduce)

The dataset name is derived by `run_pipeline.py` as `{role}_{arm}_{device}.csv`, where
`role` defaults to `train`, `device` defaults to `FakeBrisbane`, and
`arm = "swap" if --no-mp else "pol"`. So these exact filenames fall out of the defaults — the
only non-default knob is the labeling window (`--n-hi 15`).

```bash
cd src/mqtloader

# swap / routing arm  -- deterministic transpile, reproduces exactly
python run_pipeline.py --no-mp --n-lo 3 --n-hi 15 --device FakeBrisbane

# fidelity arm  -- noisy qiskit-aer simulation; values are close but NOT bit-identical
python run_pipeline.py        --n-lo 3 --n-hi 15 --device FakeBrisbane
```

Each run writes to `data/runs/<role>__<arm>__<device>__n3-15__<timestamp>/` and emits the joined
dataset under its derived name plus a `run_manifest.json`. Promote the dataset here to refresh:

```bash
cp data/runs/train__swap__FakeBrisbane__n3-15__*/train_swap_FakeBrisbane.csv data/datasets/
cp data/runs/train__pol__FakeBrisbane__n3-15__*/train_pol_FakeBrisbane.csv   data/datasets/
```

The swap arm reproduces exactly (routing is deterministic). The fidelity arm is a stochastic
noisy simulation, so a regeneration will be statistically equivalent but not row-for-row
identical to the hosted file.
