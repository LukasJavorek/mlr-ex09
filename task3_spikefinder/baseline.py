"""The two classical guesses the task text names, rebuilt on this split instead of
quoted, at the released 100 Hz - because that is where a classical method lives.

* the **fluorescence** itself: a rising calcium signal means recent spikes, and the
  correlation is taken at the metric's own 40 ms resolution. This one has no knob.
* the **positive part of the smoothed derivative**: spikes charge the dye, so the
  signal's *rise* is closer to a spike than its level is. One knob - the smoothing -
  and it is tuned on the validation split only, as everything else here is.

Both are scored with `metrics.correlate` against the same 40 ms count bins the network
is scored on, so the three columns of the comparison table are the same number over the
same bins of the same neurons. What this folder measures is 0.206 and 0.467 on the 34
test neurons (the task text quotes 0.21 and 0.45); both are reproduced by
`python baseline.py`, which prints the validation sweep and every split.

The per-neuron spread behind those means is enormous - the derivative baseline runs
from 0.15 to 0.76 across the 34 test neurons - which is why the mean is quoted with a
bootstrap interval and never on its own.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

import metrics
import sf_data

BIN = sf_data.BIN                                  # released samples per 40 ms metric bin
SIGMAS = [1, 2, 3, 5, 8, 12, 20, 30]               # Gaussian sigma in released samples = 10-300 ms


def traces(base: Path | None = None) -> dict:
    """The real part of every calcium column, at the released 100 Hz. Cached next to
    the decimated arrays; only the baselines and one figure need it, so it is kept out
    of the loader's own cache (which would be four times the size for their sake)."""
    cache = sf_data.DATA_DIR / "raw100.npz"
    if cache.exists():
        with np.load(cache) as f:
            return {(int(k.split("|")[1]), int(k.split("|")[2])): f[k] for k in f.files}
    base = base or sf_data.ensure_spikefinder()
    out = {}
    for ds in sf_data.DATASETS:
        c = pd.read_csv(base / f"{ds}.train.calcium.csv").to_numpy(np.float32)
        for j in range(c.shape[1]):
            out[(ds, j)] = c[: int(np.isfinite(c[:, j]).sum()), j]
    sf_data.DATA_DIR.mkdir(exist_ok=True)
    np.savez(cache, **{f"x|{k[0]}|{k[1]}": v for k, v in out.items()})
    return out


def in_bins(x: np.ndarray, k: int = BIN) -> np.ndarray:
    """The means of a 100 Hz trace over `k` released samples, dropping the same ragged
    tail sf_data drops, so a baseline's bins and the network's bins are the same
    intervals of time. k = 4 is the metric's own 40 ms grid; k = cfg.decimate puts the
    baseline on whatever grid a given run works on."""
    n = len(x) - len(x) % k
    return x[:n].reshape(-1, k).mean(1)


def fluorescence(x: np.ndarray, k: int = BIN) -> np.ndarray:
    """Guess 1: the level of the dye."""
    return in_bins(x, k)


def derivative(x: np.ndarray, sigma: int, k: int = BIN) -> np.ndarray:
    """Guess 2: the rise of the dye. Gaussian smoothing first - the raw difference of a
    100 Hz trace is mostly noise - then the positive part, then the means. The clip
    comes before the binning, which is what the classical recipe says and is *not* the
    same operation as clipping after it. The smoothing stays in released samples
    whatever k is: the classical detector is a 100 Hz method, and only its output grid
    follows the run."""
    s = gaussian_filter1d(x.astype(np.float64), sigma, mode="nearest")
    return in_bins(np.maximum(np.diff(s, prepend=s[0]), 0.0), k)


def predictions(raw: dict, sigma: int, k: int = BIN) -> dict:
    """Both baselines for all 174 neurons, on the run's grid."""
    return {n: {"fluorescence": fluorescence(x, k), "derivative": derivative(x, sigma, k)}
            for n, x in raw.items()}


def score_split(data: dict, preds: dict, split: str, which: str) -> dict:
    """Score one baseline over one split's neurons, against the loader's own count bins.
    `metric_bins` rather than 1 so a run on a finer grid is still judged at 40 ms."""
    truth, bins = data["y"], data["cfg"].metric_bins
    return metrics.score({n: metrics.correlate(preds[n][which], truth[n], bins)
                          for n in data["split"][split]})


def tune(data: dict, sigmas=SIGMAS, verbose: bool = True) -> int:
    """The smoothing width, chosen on the validation split alone. The curve is flat to
    the third decimal over 3-12 samples and the 18 validation neurons cannot tell the
    widths apart (the audit re-tuned it on bootstrap resamples of those 18 and the
    winner changed in a third of the draws), so this returns the argmax and the print
    says how little that argmax is worth."""
    raw = traces()
    k, bins, table = data["cfg"].decimate, data["cfg"].metric_bins, {}
    for s in sigmas:
        preds = {n: derivative(raw[n], s, k) for n in data["split"]["valid"]}
        values = [metrics.correlate(preds[n], data["y"][n], bins) for n in preds]
        table[s] = float(np.mean([v for v in values if v is not None]))
    best = max(table, key=table.get)
    if verbose:
        print(f"smoothing width, tuned on the {len(data['split']['valid'])} validation neurons:")
        for s in sigmas:
            print(f"   sigma {s:>2} samples = {s * 10:>3} ms   validation correlation "
                  f"{table[s]:.3f}{'   <-- chosen' if s == best else ''}")
        spread = max(table.values()) - min(table.values())
        print(f"   the whole curve spans {spread:.3f} and its top two widths differ by "
              f"{sorted(table.values())[-1] - sorted(table.values())[-2]:.4f};\n"
              f"   one test neuron is worth about 0.03, so this choice is nearly free "
              f"of consequence")
    return int(best)


if __name__ == "__main__":
    import sys
    # `--test` is opt-in. This file *can* score the test split, but by default it does not,
    # so that "evaluate_task3.py is the only script in this folder that reads it" stays
    # true of an ordinary pass through the folder. The final evaluation reports the two
    # test rows, and prints them beside the task text's quoted 0.21 and 0.45.
    want_test = "--test" in sys.argv
    splits = ("train", "valid", "test") if want_test else ("train", "valid")
    data = sf_data.build(sf_data.configured())
    sigma = tune(data)
    raw = traces()
    preds = predictions(raw, sigma, data["cfg"].decimate)
    print(f"\nboth baselines, sigma = {sigma} samples ({sigma * 10} ms), on "
          f"{', '.join(splits)}:")
    for split in splits:
        for which in ("fluorescence", "derivative"):
            print("   " + metrics.summarize(f"{which:<12} {split}",
                                            score_split(data, preds, split, which)))
    print("\nthe test split, from evaluate_task3.py: fluorescence 0.206, derivative 0.467;"
          " the task text quotes 0.21 and 0.45."
          + ("" if want_test else "   [pass --test to score it here instead]"))
