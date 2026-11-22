"""Task 1 data: LUDB lead II, read the way the task's data cell reads it, with
the two fixes the audit of the data called for.

What is different from the notebook's cell, and why
---------------------------------------------------
1. **The parser repairs broken waves instead of losing them.** LUDB writes a
   wave as the triplet ``'(' peak ')'``. In records 95, 104 and 111 the ``'('``
   is missing, and the strict triplet test of the original cell silently drops
   those 14 QRS complexes (record 104 loses all nine of them). `read_waves`
   keeps every annotated peak and puts a window of that class's median width in
   the same record on it.
2. **Option B: every record is cropped to its annotated span.** LUDB annotates
   only the middle ~7.3 s of each 10 s record, so 26.9% of the samples carry a
   "background" label that nobody gave: 413 real heartbeats (about one in five)
   sit outside the span. Training on them teaches the network to miss beats, and
   scoring on them punishes a model for finding them (the ceiling they imply is
   P 0.87 / QRS 0.90 / T 0.89, i.e. the quoted 0.89 QRS baseline is already at
   it). Option B keeps only the span the cardiologists actually marked:
   ``x[first_onset : last_offset]``, re-embedded left-aligned in an array of the
   original length so the shapes in the specification stay ``(1, 5000)`` in /
   ``(4, 5000)`` out. Everything dropped - the cropped edges and the padding
   that restores the length - carries the target ``-1`` and is skipped by
   ``nn.CrossEntropyLoss(ignore_index=-1)`` and by the metric.
3. **The standardization statistics are computed on the span** (see
   `Config.standardize`), because 13 records have a large amplitude artefact in
   their last 0.4 s, i.e. exactly in the cropped-away part; scaling on the whole
   record divides every real beat in those records by the artefact.

The split is the notebook's own: ``np.random.default_rng(0).permutation(200)``,
140 / 30 / 30 records, one record = one patient.
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
import wfdb
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

FS = 500                                   # Hz, one sample = 2 ms
N_REC = 200
LENGTH = 5000                              # samples per record = 10 s
IGNORE = -1                                # the target of a sample nobody annotated
WAVES = ["background", "P", "QRS", "T"]
CLASSES = (1, 2, 3)                        # the waves that are scored
wave_code = {"p": 1, "N": 2, "t": 3}
MED_WIDTH_MS = {1: 112.0, 2: 94.0, 3: 182.0}    # median width of each class over the whole base


# --------------------------------------------------------------------------- #
# the database itself
# --------------------------------------------------------------------------- #
def ensure_ludb(root: Path = DATA_DIR) -> Path:
    """The LUDB archive, downloaded and unpacked on first use (15 MB)."""
    root.mkdir(parents=True, exist_ok=True)
    base = root / "lobachevsky-university-electrocardiography-database-1.0.1"
    if len(list((base / "data").glob("*.hea"))) != N_REC or not (base / "ludb.csv").exists():
        url = ("https://physionet.org/static/published-projects/ludb/"
               "lobachevsky-university-electrocardiography-database-1.0.1.zip")
        zip_path = root / "ludb.zip"
        print(f"downloading {url}\n            -> {zip_path}")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:          # unpacked into its own new directory
            names = zf.namelist()
            assert all(not (n.startswith("/") or ".." in Path(n).parts) for n in names)
            zf.extractall(root)
        zip_path.unlink()
    return base


# --------------------------------------------------------------------------- #
# one record: the signal and its waves
# --------------------------------------------------------------------------- #
def read_waves(path: str):
    """The waves of one record as (class, onset, peak, offset) plus how many had
    to be repaired. A wave whose '(' or ')' is missing is re-created as a window
    of this class's median width in this record, centred on the peak: with one
    delimiter missing, the one that survived cannot be trusted either."""
    x = wfdb.rdrecord(path, channel_names=["ii"]).p_signal[:, 0]
    ann = wfdb.rdann(path, "ii")
    syms, smp = list(ann.symbol), [int(s) for s in ann.sample]

    clean, broken, open_at = [], [], None
    for k, s in enumerate(syms):
        if s == "(":
            open_at = smp[k]
        elif s in wave_code:
            c, peak = wave_code[s], smp[k]
            offset = smp[k + 1] if k + 1 < len(syms) and syms[k + 1] == ")" else None
            (clean if open_at is not None and offset is not None else broken).append([c, open_at, peak, offset])
            open_at = None
    widths = {c: [o - on + 1 for cc, on, _, o in clean if cc == c] for c in CLASSES}
    med = {c: int(np.median(widths[c])) if widths[c] else int(MED_WIDTH_MS[c] / 1000 * FS)
           for c in CLASSES}

    waves = [(c, on, p, o) for c, on, p, o in clean]
    for c, _on, peak, _off in broken:
        half = med[c] // 2
        waves.append((c, max(0, peak - half), peak, min(len(x) - 1, peak + half)))
    waves.sort(key=lambda w: w[1])
    return x, waves, len(broken)


# --------------------------------------------------------------------------- #
# Option B: the cropped record
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """Every data choice Task 1 makes. These are the settings the pipeline starts
    from, before validate_choices.py has measured the alternatives on the
    validation split - once it has run, `configured()` returns what it chose, and
    that is what the training and the evaluation use."""
    margin: int = 0                  # samples of real signal kept around the span (unlabelled, target -1)
    standardize: str = "span"        # "span": median/std of the annotated span; "record": of the whole record

    def key(self) -> str:
        return f"margin{self.margin}_{self.standardize}"


def configured() -> Config:
    """The data configuration to build by default: what results/choices.json says,
    once validate_choices.py has written it, and these dataclass defaults until
    then. train_task1.py and evaluate_task1.py read the same file for everything
    else, so a script that just wants "the data of this run" can call this."""
    path = HERE / "results" / "choices.json"
    if path.exists():
        return Config(**json.loads(path.read_text(encoding="utf-8"))["chosen"]["data"])
    return Config()


def build(cfg: Config = Config(), base: Path | None = None, verbose: bool = True) -> dict:
    """All 200 records as two arrays of shape (200, LENGTH), plus the metadata.

    X[i]  one channel, standardized, the annotated span (plus `margin` samples of
          context on each side) left-aligned at sample 0, zeros in the padding.
    Y[i]  the class of every sample, IGNORE (-1) in the context margin, in the
          cropped-away tail and in the padding alike.
    """
    base = Path(base) if base is not None else ensure_ludb()
    cache = DATA_DIR / f"cache_{cfg.key()}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        data = {k: z[k] for k in ("X", "Y", "left", "span", "repaired", "perm")}
        data["waves"] = [list(w) for w in z["waves"]]
        data["rhythm"] = [str(s) for s in z["rhythm"]]
        data["record"] = np.arange(1, N_REC + 1)
        data["config"] = cfg
        if verbose:
            print(f"loaded {cache.name}: X {data['X'].shape}, masked {np.mean(data['Y'] < 0):.1%}")
        return _finish(data, cfg)

    # every record is rebuilt in the same order, so the arrays, the split and this
    # list of lists all share one index: record i of the base = row i of X
    X, Y, left, span, repaired, waves_all = [], [], [], [], [], []
    for rec in range(1, N_REC + 1):
        x, waves, n_repaired = read_waves(str(base / "data" / str(rec)))
        assert len(x) == LENGTH, f"record {rec} has {len(x)} samples"
        lo, hi = waves[0][1], waves[-1][3] + 1                 # the annotated span
        clo, chi = max(0, lo - cfg.margin), min(LENGTH, hi + cfg.margin)
        window = x[clo:chi]
        if len(window) > LENGTH:
            raise ValueError(f"record {rec}: span + margin is {len(window)} > {LENGTH} samples")
        stats = x[lo:hi] if cfg.standardize == "span" else x   # see the module docstring
        xs = (window - np.median(stats)) / (stats.std() + 1e-6)

        row_x = np.zeros(LENGTH, dtype=np.float32)
        row_y = np.zeros(LENGTH, dtype=np.int64)               # background, where a cardiologist looked
        row_x[:len(xs)] = xs
        for c, onset, peak, offset in waves:                   # the span, shifted to start at 0
            row_y[onset - clo:offset - clo + 1] = c
        # everything outside the span is not "background", it is unknown: the context
        # margin, the cropped-away edge beats and the padding all become -1
        row_y[:lo - clo] = IGNORE
        row_y[hi - clo:] = IGNORE
        X.append(row_x)
        Y.append(row_y)
        left.append(clo)
        span.append((lo, hi))
        repaired.append(n_repaired)
        waves_all.append(waves)

    perm = np.random.default_rng(0).permutation(N_REC)         # one record = one patient
    meta = pd.read_csv(base / "ludb.csv")                      # the rhythm of each record, as the notebook reads it
    rhythms = dict(zip(meta["ID"], meta["Rhythms"]))
    data = dict(X=np.stack(X), Y=np.stack(Y), left=np.array(left), span=np.array(span),
                repaired=np.array(repaired), perm=perm, waves=waves_all,
                record=np.arange(1, N_REC + 1),
                rhythm=[str(rhythms[r]).strip().splitlines()[0] for r in range(1, N_REC + 1)],
                config=cfg)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=data["X"], Y=data["Y"], left=data["left"], span=data["span"],
                        repaired=data["repaired"], perm=data["perm"],
                        waves=np.array(waves_all, dtype=object), rhythm=np.array(data["rhythm"]),
                        record=data["record"])
    if verbose:
        print(f"read {N_REC} records from {base}")
    return _finish(data, cfg)


def _finish(data: dict, cfg: Config) -> dict:
    perm = data["perm"]
    data["split"] = {"train": perm[:140], "valid": perm[140:170], "test": perm[170:]}
    data["seen"] = data["Y"] >= 0
    # how many samples of each row are real signal rather than padding: the span
    # plus the context margin, left-aligned at sample 0
    hi = data["span"][:, 1]
    data["valid"] = np.minimum(LENGTH, hi + cfg.margin) - data["left"]
    return data


def summary(data: dict) -> str:
    """What the loader prints, so the same numbers are in every log."""
    Y, seen = data["Y"], data["seen"]
    lines = [f"X {data['X'].shape} float32, Y {data['Y'].shape} int64, "
             f"splits " + ", ".join(f"{k} {len(v)}" for k, v in data["split"].items()),
             f"annotated span: median {int(np.median(data['span'][:, 1] - data['span'][:, 0]))} samples "
             f"({np.median(data['span'][:, 1] - data['span'][:, 0]) / FS:.1f} s) of {LENGTH}, "
             f"target -1 on {np.mean(Y < 0):.1%} of the samples "
             f"(Option B crops away the unlabelled edges)",
             f"waves repaired by the parser: {int(data['repaired'].sum())} in records "
             f"{[i + 1 for i, n in enumerate(data['repaired']) if n]}",
             "fraction of the ANNOTATED samples per class: "
             + str(dict(zip(WAVES, np.round(np.bincount(Y[seen], minlength=4) / seen.sum(), 3))))]
    for c in CLASSES:
        per_rec = (Y == c).any(axis=1)
        n_waves = sum(sum(1 for w in r if w[0] == c) for r in data["waves"])
        lines.append(f"   {WAVES[c]:>4}: {n_waves} waves in {int(per_rec.sum())} of {N_REC} records")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# torch
# --------------------------------------------------------------------------- #
class LudbSet(Dataset):
    """(one standardized channel, the class of every sample) for the given records."""

    def __init__(self, data: dict, split: str):
        idx = data["split"][split]
        self.x = torch.from_numpy(data["X"][idx])[:, None]     # (records, 1, samples)
        self.y = torch.from_numpy(data["Y"][idx])              # (records, samples)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


def sets(data: dict) -> dict:
    return {name: LudbSet(data, name) for name in ("train", "valid", "test")}


def class_weights(data: dict, power: float = 0.5) -> np.ndarray:
    """Inverse-frequency weights over the ANNOTATED samples of the TRAINING split
    only, softened by `power` (1.0 is fully inverse-frequency). Background keeps
    weight 1: it is the majority class, it only has to be outvoted, not boosted."""
    counts = np.bincount(data["Y"][data["split"]["train"]][data["seen"][data["split"]["train"]]],
                         minlength=4).astype(float)
    w = (counts[0] / counts) ** power
    w[0] = 1.0
    return w / w[1:].mean()


def seed_everything(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":                       # python ludb_data.py  -> the data report
    import sys

    cfg = Config(margin=int(sys.argv[1]) if len(sys.argv) > 1 else 0,
                 standardize=sys.argv[2] if len(sys.argv) > 2 else "span")
    data = build(cfg)
    print(json.dumps(asdict(cfg)))
    print(summary(data))
