"""The spikefinder metric, on the grid `sf_data.build()` produces.

The challenge scores one test neuron by summing the predicted rate and the true 10 ms
counts into 40 ms bins, taking a Pearson correlation, and averaging that over neurons.
On this grid one sample already *is* one 40 ms bin, so the summing is the identity and
`correlate()` below is that metric unchanged; the `bins` argument exists so the bin
size can be varied (the audit's bin-size sweep) without a second code path.

Three properties of it shape everything else in this folder:

* It is **affine invariant**. A prediction of "counts per bin" and one of "spikes per
  second" score identically, so the network may predict a rate in units of each
  neuron's own mean rate and never needs to know that rate at test time.
* It is **undefined for a constant prediction** - a neuron predicted as a flat line
  scores nothing, not zero. The challenge itself leaves this undefined; here such a
  neuron is dropped from the mean and *counted*, because a model that outputs a
  constant would otherwise disappear from the average instead of failing.
* One neuron is worth about **+-0.03** (measured in the audit by scoring the two halves
  of every recording against each other), so the mean over the 34 test neurons has a
  noise floor near **0.005**. `score()` therefore returns a bootstrap interval over
  neurons next to every mean, and nothing here calls a difference of that size a result.
"""
from __future__ import annotations

import numpy as np

MIN_BINS = 10          # fewer points than this and a correlation is not a measurement


def correlate(pred, truth, bins: int = 1) -> float | None:
    """Pearson correlation between a predicted rate and the true counts, both on the
    loader's grid, summed into `bins` bins first (bins=1 is the challenge's 40 ms).
    None when there are too few bins, or when either side is constant."""
    if abs(len(pred) - len(truth)) > bins:
        # a prediction on a different grid than the truth would score silently and
        # meaninglessly, so every honest use of this function passes equal lengths
        raise ValueError(f"prediction ({len(pred)} samples) and truth ({len(truth)} samples) "
                         f"are on different grids")
    n = min(len(pred), len(truth))
    n -= n % bins
    if n < MIN_BINS * bins:
        return None
    p = np.asarray(pred[:n], dtype=np.float64).reshape(-1, bins).sum(1)
    t = np.asarray(truth[:n], dtype=np.float64).reshape(-1, bins).sum(1)
    if not (np.isfinite(p).all() and np.isfinite(t).all()):
        raise ValueError("the metric wants finite arrays; NaN padding never reaches here")
    if p.std() == 0 or t.std() == 0:
        return None                                  # constant side: the metric is undefined
    return float(np.corrcoef(p, t)[0, 1])


def score(per_neuron: dict) -> dict:
    """Summarize {neuron: correlation-or-None}. The correlations always come out on the
    challenge's 40 ms bins: `correlate` does the binning, so whatever grid a run worked
    on, the number this averages is already the official one.

    Two means are reported because they are different numbers: the challenge averages
    over *neurons*, which weights the datasets by how many neurons they contribute
    (dataset 8 has 40, datasets 1/3/6/7 have 4 or 5), while averaging over the *ten
    datasets* weights a session with one test neuron as much as one with twelve. For
    the derivative baseline they are 0.467 and 0.445.
    """
    got = {k: v for k, v in per_neuron.items() if v is not None}
    by_ds: dict = {}
    for (ds, _), v in got.items():
        by_ds.setdefault(ds, []).append(v)
    means = {ds: float(np.mean(v)) for ds, v in sorted(by_ds.items())}
    values = np.array(list(got.values()), dtype=np.float64)
    return {
        "bins_ms": 40,
        "mean_over_neurons": float(values.mean()) if len(values) else float("nan"),
        "mean_over_datasets": float(np.mean(list(means.values()))) if means else float("nan"),
        "n_scored": len(values),
        "n_undefined": len(per_neuron) - len(values),
        "median_neuron": float(np.median(values)) if len(values) else float("nan"),
        "min_neuron": float(values.min()) if len(values) else float("nan"),
        "max_neuron": float(values.max()) if len(values) else float("nan"),
        "per_dataset": {str(k): round(v, 4) for k, v in means.items()},
        "n_per_dataset": {str(k): len(v) for k, v in sorted(by_ds.items())},
        "boot95_over_neurons": [round(x, 4) for x in bootstrap_ci(values)],
    }


def bootstrap_ci(values: np.ndarray, draws: int = 2000, seed: int = 0) -> tuple:
    """Percentiles 2.5/97.5 of the mean, resampling the neurons with replacement - the
    honest width of a mean over this few recordings, and the yardstick every
    comparison in RESULTS.md is read against."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws_ = rng.choice(values, size=(draws, len(values)), replace=True).mean(1)
    return tuple(np.percentile(draws_, [2.5, 97.5]).tolist())


def time_shifted(preds: dict, truths: dict, seed: int = 0, bins: int = 1) -> dict:
    """The primary control: the same prediction scored against the same neuron's spike
    train, shifted by at least a third of the recording. Everything inside each signal
    survives - the autocorrelation, the burstiness, the empty bins - and only the
    alignment is destroyed, so this must come out at about zero. Measured on 12 random
    initializations of the network: -0.003 +- 0.006 on the validation split."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, p in preds.items():
        t = truths[k]
        n = min(len(p), len(t))
        out[k] = correlate(np.roll(p[:n], int(rng.integers(max(n // 3, 1), n))), t[:n], bins)
    return out


def shuffled_pairing(preds: dict, truths: dict, seed: int = 0, bins: int = 1) -> dict:
    """A second, weaker control: every prediction scored against *another* neuron's
    spike train. Read it with care - neurons recorded in the same session really do
    share signal (the audit measured that), so a prediction that is merely
    fluorescence-like can pick up a session-mate's spikes. It is reported beside
    `time_shifted` for exactly that reason: an untrained network scores a few points
    above zero here and about zero in the shifted control, which is the difference
    between "this is a filtered calcium trace" and "this predicts spikes"."""
    keys = list(preds)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))
    while np.array_equal(order, np.arange(len(keys))):      # a derangement, not a no-op
        order = rng.permutation(len(keys))
    out = {}
    for k, j in zip(keys, order):
        p, t = preds[k], truths[keys[j]]                    # different neurons, so
        n = min(len(p), len(t))                             # comparing the common
        out[k] = correlate(p[:n - n % bins], t[:n - n % bins], bins)   # head is intended
    return out


def control_means(preds: dict, truths: dict, seed: int = 0, bins: int = 1) -> dict:
    """Both controls, as means - the two numbers a score table prints under its rows."""
    def mean(d):
        vals = [v for v in d.values() if v is not None]
        return round(float(np.mean(vals)), 4) if vals else float("nan")
    return {"shifted": mean(time_shifted(preds, truths, seed, bins)),
            "shuffled": mean(shuffled_pairing(preds, truths, seed, bins))}


def summarize(name: str, s: dict) -> str:
    """One line per method, in the order every log prints them."""
    lo, hi = s["boot95_over_neurons"]
    return (f"{name:<26} mean over {s['n_scored']:>2} neurons {s['mean_over_neurons']:.3f} "
            f"[{lo:+.3f}, {hi:+.3f}]   mean over datasets {s['mean_over_datasets']:.3f}   "
            f"neurons {s['min_neuron']:+.3f} .. {s['max_neuron']:+.3f}   "
            f"undefined {s['n_undefined']}")
