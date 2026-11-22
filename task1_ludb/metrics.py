"""The metric of Task 1: the Dice of every wave, counted over the annotated
samples only.

Both the model and the baseline are scored with the same function on the same
arrays, so the two numbers mean the same thing. Samples with target -1 (the
edges Option B cropped away and the padding that restores the record length)
take part in neither the numerator nor the denominators.

Two ways of averaging are reported side by side, because they answer different
questions and 24 records have no P wave at all:
  * "global" pools the counts of every record (a big, well-labelled record then
    weighs more than a small one), and
  * "per record" is the mean over the records where the wave can be scored at all:
    one in which it is labelled, or one in which the method painted it anyway -
    which scores 0 there, because inventing a P wave in a rhythm that has none is
    an error, not a missing value. records_labelled says how many records a
    cardiologist marked the wave in, which is the smaller number when they differ.
"""
from __future__ import annotations

import numpy as np

from ludb_data import CLASSES, WAVES, IGNORE


def dice_counts(pred: np.ndarray, target: np.ndarray, c: int):
    """(true positives, predicted samples, labelled samples) of class c, over the
    annotated samples of one record."""
    seen = target >= 0
    p, t = (pred == c) & seen, (target == c) & seen
    return int((p & t).sum()), int(p.sum()), int(t.sum())


def dice(pred: np.ndarray, target: np.ndarray, c: int):
    """Dice of class c for one record, or None where it cannot be said (the class
    is neither labelled nor predicted there)."""
    tp, n_pred, n_true = dice_counts(pred, target, c)
    if n_pred + n_true == 0:
        return None
    return 2 * tp / (n_pred + n_true)


def evaluate(pred: np.ndarray, target: np.ndarray, classes=CLASSES) -> dict:
    """{"QRS": {"global": …, "per_record": …, "records": n}, …} for a whole split."""
    out = {}
    for c in classes:
        tp = n_pred = n_true = 0
        per_record = []
        labelled = 0
        for p, t in zip(pred, target):
            a, b, n = dice_counts(p, t, c)
            tp, n_pred, n_true = tp + a, n_pred + b, n_true + n
            labelled += n > 0
            d = dice(p, t, c)
            if d is not None:
                per_record.append(d)
        global_dice = 2 * tp / (n_pred + n_true) if n_pred + n_true else float("nan")
        out[WAVES[c]] = {"global": float(global_dice),
                         "per_record": float(np.mean(per_record)) if per_record else float("nan"),
                         "records": len(per_record),
                         # how many of these records a cardiologist actually marked this
                         # wave in - "records" counts the ones where a Dice could be said
                         # at all, which includes a record that has no P but where the
                         # method painted one anyway (there its Dice is 0, not missing)
                         "records_labelled": int(labelled),
                         "tp": tp, "pred": n_pred, "labelled": n_true,
                         # Dice alone hides which half is wrong; these two say whether a
                         # method misses waves (low recall) or invents them (low precision)
                         "recall": tp / n_true if n_true else float("nan"),
                         "precision": tp / n_pred if n_pred else float("nan")}
    return out


def balance_report(scores: dict) -> str:
    """What the Dice of a class is made of: how many samples were labelled, predicted
    and hit, and hence whether the method misses waves or invents them."""
    lines = ["                 labelled  predicted      hit   recall  precision      Dice"]
    for name, s in scores.items():
        lines.append(f"  {name:<9}{s['labelled']:>9,} {s['pred']:>10,} {s['tp']:>8,}"
                     f"   {s['recall']:>6.3f}    {s['precision']:>6.3f}   {s['global']:>7.3f}")
    return "\n".join(lines)


def runs(mask: np.ndarray):
    """The [start, end] sample ranges of every run of True in a 1-D mask."""
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return list(zip(idx[0::2], idx[1::2] - 1))


def wave_durations_ms(pred: np.ndarray, target: np.ndarray, c: int, fs: int):
    """(annotated widths in ms, predicted widths in ms) of the waves of class c in
    one record - the QRS duration of one patient, measured both ways."""
    seen = target >= 0
    true_runs = [r for r in runs((target == c) & seen)]
    pred_runs = [r for r in runs((pred == c) & seen)]
    to_ms = lambda rr: [round((b - a + 1) * 1000 / fs, 1) for a, b in rr]
    return to_ms(true_runs), to_ms(pred_runs)


def matched_wave_durations_ms(pred, target, c, fs, tolerance_ms=60.0):
    """For every annotated wave of class c, the predicted wave that overlaps it
    (if any): a list of (annotated ms, predicted ms) pairs. Comparing a wave to
    the wave it actually is beats comparing two lists of different lengths."""
    seen = target >= 0
    tol = tolerance_ms * fs / 1000
    pairs = []
    for a, b in runs((target == c) & seen):
        best, best_overlap = None, 0
        for pa, pb in runs((pred == c) & seen):
            overlap = min(b, pb) - max(a, pa) + 1
            if overlap > best_overlap and abs(pa - a) <= tol and abs(pb - b) <= tol:
                best, best_overlap = (pa, pb), overlap
        if best:
            width = lambda r: round((r[1] - r[0] + 1) * 1000 / fs, 1)
            pairs.append((width((a, b)), width(best)))
    return pairs


def table(rows: dict, classes=CLASSES) -> str:
    """A printed table of Dice, one row per method. rows = {name: evaluate(...)}."""
    head = " " * 36 + "".join(f"{WAVES[c]:>16}" for c in classes)
    lines = [head]
    for name, scores in rows.items():
        cells = ""
        for c in classes:
            s = scores[WAVES[c]]
            per = "     -     " if np.isnan(s["per_record"]) else f"({s['per_record']:.3f})"
            cells += f"{s['global']:>10.3f} {per}"
        lines.append(f"{name:<36}{cells}")
    lines.append("  global Dice over the annotated samples; in brackets the mean over the records")
    lines.append("  where that wave was scored at all - a method that paints a wave nobody")
    lines.append("  labelled in that record is scored 0 there, not skipped (see metrics.py)")
    return "\n".join(lines)


def sanity_check() -> None:                       # python metrics.py
    fs = 500
    t = np.full(20, IGNORE, dtype=np.int64)                 # a record: 14 annotated samples,
    t[:14] = 0                                              # 6 samples Option B cropped away
    t[2:6] = 2                                              # one QRS, 4 samples = 8 ms
    t[9:13] = 3                                             # one T wave
    perfect = np.where(t >= 0, t, 0)                        # argmax can never output -1
    outside = perfect.copy()
    outside[15:19] = 2                                      # a false beat in the cropped-away part
    late = perfect.copy()
    late[2] = 0
    late[6] = 2                                             # the QRS predicted one sample late
    print("perfect prediction :", evaluate(perfect, t), "\n")
    assert dice(perfect, t, 2) == 1.0 and dice(perfect, t, 3) == 1.0
    assert dice(outside, t, 2) == 1.0, "a false beat outside the annotated span must not count"
    q = evaluate(late, t)["QRS"]
    print("QRS one sample late:", q)
    assert abs(q["global"] - 2 * 3 / (4 + 4)) < 1e-9, q     # TP 3, of 4 predicted and 4 labelled
    assert q["labelled"] == 4 and q["pred"] == 4 and q["tp"] == 3
    assert dice(late[..., :14], t[..., :14], 2) == dice(late, t, 2), "the padding must not matter"
    assert evaluate(np.where(t >= 0, 3, 0), t)["P"]["records"] == 0, "no P anywhere: P is not scored"
    assert evaluate(np.where(t >= 0, 0, 0), t)["QRS"]["global"] == 0.0, "a model that finds nothing scores 0"
    print("runs of the QRS   :", runs(late == 2))
    print("durations (annotated, predicted):", wave_durations_ms(late, t, 2, fs))
    print("matched waves (annotated ms, predicted ms):", matched_wave_durations_ms(late, t, 2, fs))
    print("ok")


if __name__ == "__main__":
    sanity_check()
