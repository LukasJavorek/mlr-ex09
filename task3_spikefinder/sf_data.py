"""Task 3 data: the spikefinder training release, read the way the task's data cell
reads it, with the four changes the audit of the data (`../task3_data_audit.md`)
called for.

What is different from the notebook's cell, and why
----------------------------------------------------
1. **The 100 Hz grid is averaged down to 25 Hz.** 86-100% of every trace's power is
   below 5 Hz and at most 0.7% of it is above 25 Hz (a spike train measured the same
   way has 45% above 25 Hz, so the method is not blind), i.e. three of the four
   samples a 40 ms metric bin is made of are copies of their neighbour. Averaging 4
   samples into 1 loses nothing measurable, costs 4x less compute, and gives the
   *same* U-Net a receptive field of 1.76 s instead of 0.44 s - which matters,
   because the fluorescence takes 0.4-2.3 s to fall back to half its peak after a
   spike. The input is a mean, the target a **sum**: counts per 40 ms bin, the
   metric's own resolution.
2. **Every neuron is standardized on its own trace.** The std of the fluorescence
   spans a factor of 13 and the firing rate a factor of 793 between neurons. The
   statistics come from that neuron's fluorescence only - never from its spikes - so
   this is not leakage: at evaluation time the trace being normalized is the trace
   being predicted.
3. **Windows never contain padding.** 9% of the cells in the ten tables are NaN at
   the end of a column (80 of the 174 columns have some). The window list is built
   per neuron from `np.isfinite(trace).sum()`, non-overlapping and never crossing
   the edge, so no window is partly fabricated and no NaN reaches BatchNorm.
   Shortest recording: 55 s, so every neuron yields at least two windows of 20.5 s.
4. **The duplicate is dropped from training.** Dataset 9 holds one neuron twice:
   columns 3 and 4 differ by at most 1.2e-05 and their spike columns are identical in
   100% of the samples, and the notebook's split puts one copy in train and the other
   in test. `(9, 3)` is removed from the training set (1 of 122 neurons, no test
   neuron changes), which closes the only leak the per-dataset split cannot see.

The target of a window is the count vector **divided by that neuron's own mean count
per bin**, so the network predicts a *relative* rate. The metric is a Pearson
correlation - invariant to the scale of the prediction - so the neuron's mean rate is
never needed at evaluation time, and dividing by it stops the 12 spikes/s neurons
from outweighing the 0.02 spikes/s ones in the loss (an unweighted MSE over
concatenated windows gives 10 of the 174 neurons 30% of its weight; the score weights
all 174 the same).

The split is the notebook's own, rebuilt line for line: `np.random.default_rng(0)`,
`rng.permutation` per dataset, `n_test = max(1, round(0.2n))`,
`n_val = max(1, round(0.1n))` - 122 / 18 / 34 neurons, of which this file drops one
from training as above.

Nothing here reads the test neurons' spike columns for anything but the score: the
test columns are loaded (they are in the same tables) and only `evaluate_task3.py`
looks at them.
"""
from __future__ import annotations

import json
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

SF_URL = "https://s3.amazonaws.com/neuro.datasets/challenges/spikefinder/spikefinder.train.zip"
DATASETS = list(range(1, 11))
FS_IN = 100                                  # Hz, as released: one sample = 10 ms
BIN = 4                                      # released samples per 40 ms metric bin
TWIN = (9, 3)                                # the training copy of dataset 9's duplicated neuron
MU_FLOOR = 1e-3                              # counts per bin; keeps the quietest neuron's
                                             # relative target from exploding (see summary())


# --------------------------------------------------------------------------- #
# the archive itself
# --------------------------------------------------------------------------- #
def ensure_spikefinder(root: Path = DATA_DIR) -> Path:
    """The challenge's training archive, downloaded and unpacked on first use (35 MB)."""
    root.mkdir(parents=True, exist_ok=True)
    base = root / "spikefinder.train"
    if len(list(base.glob("*.train.*.csv"))) != 20:
        zip_path = root / "spikefinder.train.zip"
        if not zip_path.exists():
            print(f"downloading {SF_URL}\n            -> {zip_path}")
            urllib.request.urlretrieve(SF_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:          # unpacked into its own new directory
            names = zf.namelist()
            assert all(not (n.startswith("/") or ".." in Path(n).parts) for n in names), \
                "the archive contains a path that would escape its directory"
            zf.extractall(root)
    return base


# --------------------------------------------------------------------------- #
# every data choice Task 3 makes
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """The settings the pipeline starts from. Once validate_choices.py has run,
    `configured()` returns what it measured on the validation split, and that is
    what the training and the evaluation use."""
    decimate: int = BIN                      # 4: 100 Hz -> 25 Hz, one input sample per metric bin
    window: int = 512                        # input samples per window: 512/25 = 20.5 s
    standardize: str = "neuron"              # "neuron": z-scored on its own trace; "none": as released
    scale: str = "neuron"                    # "neuron": target / that neuron's mean rate; "global": / one
                                             # shared rate - the arm that tests the weighting argument
    drop_duplicate: bool = True              # keep (9, 3) out of the training set
    seed: int = 0                            # the notebook's own seed for the split

    def key(self) -> str:
        # the cache holds x and y, which depend only on these two: window, scale and
        # drop_duplicate are re-derived from the Config on every load, so a sweep over
        # window sizes reads the csv files once instead of once per arm
        return f"d{self.decimate}_{self.standardize}"

    @property
    def fs(self) -> int:
        return FS_IN // self.decimate

    @property
    def seconds(self) -> float:
        return self.window / self.fs

    @property
    def metric_bins(self) -> int:
        """How many input samples make one 40 ms metric bin on this grid - 1 at 25 Hz,
        2 at 50 Hz, 4 at 100 Hz. `metrics.correlate` always uses it, so every arm of
        every sweep is scored on the challenge's 40 ms bins and nothing else, however
        coarsely or finely it represents the signal."""
        return max(1, BIN // self.decimate)


def configured() -> Config:
    """The data configuration of this run: results/choices.json if it exists, the
    dataclass defaults until it does."""
    path = HERE / "results" / "choices.json"
    if path.exists():
        return Config(**json.loads(path.read_text(encoding="utf-8"))["chosen"]["data"])
    return Config()


# --------------------------------------------------------------------------- #
# reading, decimating, splitting
# --------------------------------------------------------------------------- #
def split_of(neurons, seed: int = 0) -> dict:
    """The notebook's split, rebuilt exactly: the same rng, the same order of
    operations, so these are the train/valid/test neurons the exercise fixed."""
    rng = np.random.default_rng(seed)
    out = {"train": [], "valid": [], "test": []}
    for ds in DATASETS:
        idx = rng.permutation([k for k, n in enumerate(neurons) if n[0] == ds])
        n_test, n_val = max(1, round(0.2 * len(idx))), max(1, round(0.1 * len(idx)))
        out["test"] += [neurons[k] for k in idx[:n_test]]
        out["valid"] += [neurons[k] for k in idx[n_test:n_test + n_val]]
        out["train"] += [neurons[k] for k in idx[n_test + n_val:]]
    return out


def decimate(trace: np.ndarray, counts: np.ndarray, k: int):
    """The real part of one neuron on the coarse grid: the input averaged over k
    samples (a box whose first null is exactly at the new Nyquist frequency), the
    target summed over the same k samples. A tail shorter than k is dropped from
    both, so bin b of the input and bin b of the target are always the same 40 ms."""
    n = int(np.isfinite(trace).sum())
    assert np.isfinite(counts[:n]).all() and not np.isfinite(counts[n:]).any(), \
        "calcium and spikes must end at the same sample"
    n -= n % k
    return (trace[:n].reshape(-1, k).mean(1).astype(np.float32),
            counts[:n].reshape(-1, k).sum(1).astype(np.float32))


def window_starts(length: int, window: int) -> np.ndarray:
    """Non-overlapping window starts, none of which reaches past the real signal."""
    return np.arange(0, max(0, length - window + 1), window, dtype=np.int64)


def build(cfg: Config = Config(), base: Path | None = None, verbose: bool = True) -> dict:
    """All 174 neurons as per-neuron arrays on the coarse grid, the notebook's split,
    the window list and the target scales. The decimated arrays are cached in data/
    under a name that encodes the Config; the split and the windows are recomputed
    from it every time, so a wrong cache cannot quietly survive a change of seed."""
    base = base or ensure_spikefinder()
    cache = DATA_DIR / f"cache_{cfg.key()}.npz"
    if cache.exists():
        with np.load(cache) as loaded:
            x = {(int(n.split("|")[1]), int(n.split("|")[2])): loaded[n]
                 for n in loaded.files if n.startswith("x|")}
            y = {(int(n.split("|")[1]), int(n.split("|")[2])): loaded[n]
                 for n in loaded.files if n.startswith("y|")}
        assert sorted(x) == sorted(y) and len(x) == 174, f"{cache.name} is not complete"
        out = _assemble(cfg, sorted(x), x, y, source=f"loaded {cache.name}")
    else:
        calcium = {ds: pd.read_csv(base / f"{ds}.train.calcium.csv").to_numpy(np.float32)
                   for ds in DATASETS}
        spikes = {ds: pd.read_csv(base / f"{ds}.train.spikes.csv").to_numpy(np.float32)
                  for ds in DATASETS}
        for ds in DATASETS:                    # the two tables must describe one recording
            assert calcium[ds].shape == spikes[ds].shape, f"dataset {ds}: shapes differ"
            assert (np.isfinite(calcium[ds]) == np.isfinite(spikes[ds])).all(), \
                f"dataset {ds}: the NaN masks differ"
        neurons = [(ds, j) for ds in DATASETS for j in range(calcium[ds].shape[1])]
        assert len(neurons) == 174, f"expected 174 neurons, found {len(neurons)}"

        x, y = {}, {}
        for ds, j in neurons:
            c, s = decimate(calcium[ds][:, j], spikes[ds][:, j], cfg.decimate)
            y[(ds, j)] = s
            # the standardization statistics of a neuron come from its own trace, and
            # from the trace only - the counts are never looked at here
            x[(ds, j)] = (c - c.mean()) / (c.std() + 1e-6) if cfg.standardize == "neuron" else c
        DATA_DIR.mkdir(exist_ok=True)
        np.savez(cache, **{f"x|{k[0]}|{k[1]}": v for k, v in x.items()},
                 **{f"y|{k[0]}|{k[1]}": v for k, v in y.items()})
        out = _assemble(cfg, neurons, x, y, source=f"read 20 csv from {base}, cached -> {cache.name}")

    out["cfg"] = cfg
    if verbose:
        print("data: " + out["source"])
        print(summary(out))
    return out


def _assemble(cfg: Config, neurons, x: dict, y: dict, source: str) -> dict:
    """Everything that follows from the arrays: the split, the windows, and the
    per-neuron scale the target is divided by."""
    split = split_of(neurons, cfg.seed)
    if cfg.drop_duplicate and TWIN in split["train"]:
        split["train"].remove(TWIN)                      # the audit's one-line fix

    starts = {n: window_starts(len(x[n]), cfg.window) for n in neurons}
    assert all(len(s) >= 1 for s in starts.values()), \
        "a neuron is shorter than one window; the window is too long for this data"

    # The scale of the target, from the TRAINING windows of that neuron only (the
    # validation ones too, since their labels are legitimately available there). A
    # test neuron has none: the metric cannot see a scale, so it never needs one.
    mu = {}
    for n in split["train"] + split["valid"]:
        bins = np.concatenate([y[n][a:a + cfg.window] for a in starts[n]])
        mu[n] = float(max(bins.mean(), MU_FLOOR))
    if cfg.scale == "global":
        # one rate for every neuron: the same Poisson loss, but the loud neurons now
        # carry their real weight in it. This is the arm that measures what dividing by
        # mu was worth; "neuron" is the default and what the spec commits to.
        pooled = float(max(np.concatenate([y[n][a:a + cfg.window]
                                           for n in split["train"]
                                           for a in starts[n]]).mean(), MU_FLOOR))
        mu = {n: pooled for n in mu}
    return {"source": source, "neurons": neurons, "split": split, "x": x, "y": y,
            "mu": mu, "starts": starts,
            "n_floored": sum(v == MU_FLOOR for v in mu.values())}


def summary(data: dict) -> str:
    """What the loader prints, so the same numbers are in every log."""
    cfg, n = data["cfg"], {k: len(v) for k, v in data["split"].items()}
    samples = sum(len(v) for v in data["x"].values())
    empty = np.concatenate([(v == 0) for v in data["y"].values()]).mean()
    windows = {k: sum(len(data["starts"][m]) for m in v) for k, v in data["split"].items()}
    rate = list(data["mu"].values())
    unfloored = [r for r in rate if r > MU_FLOOR]     # the floor is a guard, not a measurement
    return "\n".join([
        f"{len(data['neurons'])} neurons on a {cfg.fs} Hz grid ({cfg.decimate} released samples per bin): "
        f"{samples / cfg.fs:.0f} s = {samples / cfg.fs / 3600:.1f} h of recording",
        f"input (channels, window) = (1, {cfg.window}) = {cfg.seconds:.1f} s, "
        f"standardized per neuron: {cfg.standardize == 'neuron'}",
        "splits " + ", ".join(f"{k} {v} neurons / {windows[k]} windows" for k, v in n.items())
        + ("" if not cfg.drop_duplicate else "   [(9,3), the duplicate's training copy, dropped]"),
        f"empty bins ({cfg.decimate * 10} ms each): {empty:.1%}; the target is a spike count, "
        f"0-6 per 40 ms",
        f"shortest neuron {min(len(v) for v in data['x'].values()) / cfg.fs:.0f} s, "
        f"longest {max(len(v) for v in data['x'].values()) / cfg.fs:.0f} s, "
        f"tail dropped per neuron < {cfg.window / cfg.fs:.1f} s",
        f"target scale ({cfg.scale} mean count per bin): "
        f"{min(rate):.4f} .. {max(rate):.4f} over the {len(rate)} train+valid neurons, "
        f"a factor of {max(unfloored) / min(unfloored):.0f} between the quietest and loudest "
        f"neuron that is not at the floor"
        + (f"  [{data['n_floored']} neuron sits at the {MU_FLOOR:g} floor, counted in the range "
           f"but not in that factor]" if data["n_floored"] else "")])


# --------------------------------------------------------------------------- #
# torch
# --------------------------------------------------------------------------- #
class SpikeSet(Dataset):
    """(standardized fluorescence, count per bin in units of the neuron's own mean
    rate) for one window. Both are (1, window); the target is float32 because the
    Poisson loss wants a rate, not a class index."""

    def __init__(self, data: dict, split: str):
        cfg = data["cfg"]
        xs, ys = [], []
        for key in data["split"][split]:
            x, y, scale = data["x"][key], data["y"][key], data["mu"].get(key, 1.0)
            for a in data["starts"][key]:
                xs.append(x[a:a + cfg.window])
                ys.append(y[a:a + cfg.window] / scale)
        self.x = torch.from_numpy(np.stack(xs).astype(np.float32))[:, None]   # (windows, 1, window)
        self.y = torch.from_numpy(np.stack(ys).astype(np.float32))[:, None]

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


def sets(data: dict, splits=("train", "valid", "test")) -> dict:
    """The torch datasets, also attached to `data` as `train_set`, `valid_set`,
    `test_set` - every window of every neuron of a split, stacked: the whole training
    split is ~1700 windows of 512 floats, a few megabytes, so it lives in RAM and no
    worker process ever has to read a file.

    `validate_choices.py` asks for `("train", "valid")` only. Nothing in the sweep can
    score, average or threshold a test window if no test window is materialized, and
    the loader's own target arrays stay untouched either way - what separates the two
    scripts is not which bytes they read but what they are allowed to compute from
    them."""
    for name in splits:
        data[f"{name}_set"] = SpikeSet(data, name)
    return {name: data[f"{name}_set"] for name in splits}


def seed_everything(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":                    # python sf_data.py [decimate] [window]
    import sys

    cfg = Config(decimate=int(sys.argv[1]) if len(sys.argv) > 1 else BIN,
                 window=int(sys.argv[2]) if len(sys.argv) > 2 else 512)
    print(json.dumps(asdict(cfg)))
    build(cfg)
