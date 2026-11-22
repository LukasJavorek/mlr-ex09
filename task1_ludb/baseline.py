"""The baseline the task defines, implemented in full.

"R peaks with `find_peaks` on the band-passed signal, then P, QRS and T placed at
their median distances from the R peak, those distances measured on the training
patients."

Two parts, and only the second one is learned:
  1. `r_peaks` - band-pass 5-15 Hz, find the peaks, keep the ones tall enough to
     be a heartbeat (a threshold relative to the record's own median prominence,
     because the records differ in amplitude by a factor of three even after
     standardization).
  2. `measure_offsets` - on the TRAINING records only, for every labelled wave,
     the distance from its peak to the peak of the nearest QRS, and of its onset
     and offset to that same R peak. The medians of those distances say where a
     P wave sits relative to a heartbeat. Nothing from the validation or the test
     records enters them.

The detector's two thresholds are the only thing tuned on the validation split
(`sweep` in validate_choices.py), never on the test split.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from ludb_data import CLASSES, FS, IGNORE, WAVES


@dataclass
class BaselineParams:
    """The three thresholds of the detector, chosen on the VALIDATION split by the
    mean of the three Dice over band {5-15, 8-20} Hz x distance {200, 240, 300} ms x
    keep_rel {0.2, 0.35, 0.5, 0.65, 0.8} x anchor {max, p90, p75, median} - 96
    settings, see results/choices.json. Nothing here is close to the edge of that
    grid: inside the 8-20 Hz band every setting from keep_rel 0.2 to 0.5 scores
    within 0.005 of the winner, so the baseline does not depend on its own
    thresholds - only the band does, and 8-20 Hz beats 5-15 Hz by 0.13."""
    band_hz: tuple = (8.0, 20.0)   # the band-pass that makes the QRS the loudest thing in the signal
    distance_ms: int = 240         # the shortest interval between two heartbeats (30 s < 250 bpm)
    keep_rel: float = 0.35         # a peak must reach this fraction of the record's tall peaks...
    anchor: str = "p90"            # ...measured against this statistic of their prominences


def r_peaks(x: np.ndarray, valid_len: int, p: BaselineParams = BaselineParams()) -> np.ndarray:
    """The sample indices of the R peaks of one (standardized, cropped) record."""
    seg = x[:valid_len]
    if len(seg) < 4 * FS:                              # sosfiltfilt needs a few periods
        return np.array([], dtype=int)
    sos = butter(3, p.band_hz, btype="bandpass", fs=FS, output="sos")
    bp = sosfiltfilt(sos, seg - seg.mean())
    peaks, props = find_peaks(bp, distance=int(p.distance_ms / 1000 * FS), prominence=1e-3)
    if len(peaks) == 0:
        return peaks
    # a threshold relative to the record's own peaks, because the records differ in
    # amplitude by a factor of three even after standardization
    quantile = {"max": 100.0, "p90": 90.0, "p75": 75.0, "median": 50.0}[p.anchor]
    return peaks[props["prominences"] >= p.keep_rel * np.percentile(props["prominences"], quantile)]


def measure_offsets(data: dict) -> dict:
    """The median (onset, peak, offset) distance to the nearest R peak of each
    class, measured on the TRAINING records. Distances are in samples and signed:
    a P wave is before the R peak, a T wave after it."""
    gaps = {c: {"onset": [], "peak": [], "offset": []} for c in CLASSES}
    for i in data["split"]["train"]:
        qrs = [w[2] for w in data["waves"][i] if w[0] == 2]
        if not qrs:
            continue
        for c, onset, peak, offset in data["waves"][i]:
            r = min(qrs, key=lambda q: abs(peak - q))
            for name, value in (("onset", onset), ("peak", peak), ("offset", offset)):
                gaps[c][name].append(value - r)
    return {c: {k: float(np.median(v)) for k, v in d.items()} for c, d in gaps.items()}


def predict_record(x: np.ndarray, valid_len: int, offsets: dict,
                   p: BaselineParams = BaselineParams()) -> np.ndarray:
    """The delineation of one record: background everywhere, a wave window of the
    median width and median position around every R peak. Painted in time order,
    so that a later wave overwrites an earlier one exactly like the labels do."""
    y = np.zeros(valid_len, dtype=np.int64)
    placed = []
    for r in r_peaks(x, valid_len, p):
        for c in CLASSES:                              # P, then QRS, then T
            placed.append((offsets[c]["onset"] + r, c, offsets[c]["offset"] + r))
    for onset, c, offset in sorted(placed):
        # clipped at both ends on purpose: a window that starts before the first
        # sample of the record must paint nothing, not everything up to its
        # negative end index
        a, b = max(0, int(round(onset))), min(valid_len - 1, int(round(offset)))
        if a <= b:
            y[a:b + 1] = c
    return y


def predict_all(data: dict, params: BaselineParams | None = None):
    """The baseline's delineation of all 200 records, and the offsets it used."""
    p = params or BaselineParams()
    offsets = measure_offsets(data)
    pred = np.zeros_like(data["Y"])
    for i in range(len(data["X"])):
        pred[i, :data["valid"][i]] = predict_record(data["X"][i], data["valid"][i], offsets, p)
    return pred, offsets, p


def describe_offsets(offsets: dict) -> str:
    lines = ["median distance from the R peak, in ms, measured on the training records:",
             "  class        onset       peak     offset       width"]
    for c in CLASSES:
        o = offsets[c]
        lines.append(f"  {WAVES[c]:>6} {o['onset'] * 1000 / FS:>9.0f} {o['peak'] * 1000 / FS:>9.0f} "
                     f"{o['offset'] * 1000 / FS:>9.0f} {(o['offset'] - o['onset']) * 1000 / FS:>8.0f}")
    return "\n".join(lines)


def detection_report(data: dict, params: BaselineParams | None = None, split: str = "valid") -> str:
    """How well the R peak detector alone agrees with the labelled QRS peaks, on a
    split: the baseline is only as good as its beat detection, and this is the
    number that says so."""
    p = params or BaselineParams()
    hits = labelled = found = 0
    for i in data["split"][split]:
        peaks = r_peaks(data["X"][i], data["valid"][i], p)
        qrs = [w[2] - data["left"][i] for w in data["waves"][i] if w[0] == 2]   # same (cropped) axis
        labelled += len(qrs)
        found += len(peaks)
        hits += sum(any(abs(int(q) - pk) <= int(0.06 * FS) for pk in peaks) for q in qrs)
    return (f"R peaks on {split}: {hits}/{labelled} labelled QRS found "
            f"({hits / max(labelled, 1):.1%}), {found} peaks found in total")


if __name__ == "__main__":
    import json
    import sys

    from ludb_data import Config, build
    data = build(Config())
    pred, offsets, p = predict_all(data)
    print(json.dumps(asdict(p)))
    print(describe_offsets(offsets))
    print(detection_report(data, p, "train"))
    print(detection_report(data, p, "valid"))
    from metrics import evaluate, table
    for split in ("train", "valid"):                    # the test split is not touched here
        idx = data["split"][split]
        print(f"\n{split}: baseline");print(table({"baseline": evaluate(pred[idx], data["Y"][idx])}))
