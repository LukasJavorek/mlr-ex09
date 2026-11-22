"""The one place a Task 6 number is computed. Baseline, both networks, every figure.

Spec item 4 asks for exactly this file: one metric function whose convention is written in
its docstring, so the baseline, route A and route B are all scored by the same code path.

THE CONVENTIONS, in full, because `task6_data_audit.md` section 2 found two incompatible
ones in use inside this exercise and they rank the predictors in *opposite* directions:

`PSNR`
    per photo: `10*log10(1/MSE)`, the MSE over all three channels of the [0,1] images.
    Reported four ways, never one on its own:
      `psnr_mean`       mean of the per-photo values - the task's "about 25 dB" for gray
      `psnr_median`     the median, which barely moves when the convention does
      `psnr_clip40`     mean after clipping each photo at `CLIP_DB` - 8 of the 500 test
                        photos are already black and white, so their PSNR is a property of
                        `EPS` in this file (it floors at 120 dB), not of the picture
      `psnr_pooled`     `10*log10(1/mean MSE)` - the notebook's only PSNR code (Task 4,
                        cell 30). **On this data the two conventions disagree about
                        everything**: per-photo-then-mean ranks gray first (26.9 dB) and
                        every coloured answer below it, pooled ranks the conditional mean
                        first. A result quoted without saying which is not a result.
`chroma L1`
    mean over *pixels* of `(|dCb| + |dCr|) / 255` - the two channels summed. Gray scores
    0.0804, which is twice the truth's colourfulness: that factor of two is the identity
    that catches a rescaling bug.
`colourfulness`
    mean over pixels *and* channels of `|chroma| / 255`. 0.041 on the real photos, exactly
    0 for gray, which is the definition the task's "about 0.04" identifies (audit section 3
    rejects Hasler-Susstrunk, 0.138, and the mean chroma radius, 0.062, on that ground).
`hue error`
    angle between predicted and true chroma vectors in degrees, over the pixels with a hue
    to get wrong (radius > 20 of 128). Reported, never optimised: the audit shows it is
    flat at ~60 deg for every shrinkage of the optimal answer.

EVERY score is paired-tested where a comparison is claimed: `paired()` cancels the
photo-to-photo spread (per-photo sd is 13-17 dB, which is why an unpaired "+0.4 dB over the
baseline" on 100 photos means nothing - audit section 2 and 7).
"""
from __future__ import annotations

import numpy as np

import colorspace as cs

CLIP_DB = 40.0          # each photo's PSNR is clipped here before averaging (audit section 2)
EPS = 1e-12             # the floor that turns a perfect photo into 120 dB instead of infinity
YB = 64                 # luminance bins for the ceilings, as in the audit's lookup tables
DRAWS = 2000
SEED = 0


# ---------------------------------------------------------------------------------- PSNR --
def mse_photos(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-photo MSE on [0,1]. The pooled PSNR is 10*log10(1/mean of these), so a
    bootstrap over photos covers both conventions' noise."""
    if pred.shape != truth.shape:
        raise ValueError(f"{pred.shape} vs {truth.shape}")
    d = pred.astype(np.float32) - truth.astype(np.float32)
    return (d * d).mean((1, 2, 3)) / 255.0 ** 2


def psnr_photos(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-photo PSNR, unclipped. Clip with `clip_db()` when averaging; do not clip when
    counting wins against the baseline (a tie at 120 dB should not read as a win)."""
    return 10 * np.log10(1.0 / np.maximum(mse_photos(pred, truth), EPS))


def clip_db(per_photo: np.ndarray, at: float = CLIP_DB) -> np.ndarray:
    return np.minimum(per_photo, at)


def psnr_pooled(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(10 * np.log10(1.0 / max(mse_photos(pred, truth).mean(), EPS)))


# --------------------------------------------------------------------------------- score --
def score(pred_rgb: np.ndarray, truth_rgb: np.ndarray,
          baseline_per_photo: np.ndarray | None = None) -> dict:
    """Everything the specification's metric row asks for, for one set of predictions.

    `pred_rgb` is the reconstructed photo (what `colorspace.rgb_of` returns with the *input*
    Y written back), `truth_rgb` the photo itself. Both (N, 64, 64, 3), 0..255. The chroma
    of the prediction is read off the reconstructed RGB rather than from what was fed in,
    so that clipping in `rgb_of` is charged to the prediction, as it is in the picture.
    """
    _, cb_p, cr_p = cs.ycc(pred_rgb)
    _, cb_t, cr_t = cs.ycc(truth_rgb)
    per = psnr_photos(pred_rgb, truth_rgb)
    out = {
        "n": int(len(per)),
        "psnr_mean": round(float(per.mean()), 3),
        "psnr_median": round(float(np.median(per)), 3),
        "psnr_clip40": round(float(clip_db(per).mean()), 3),
        "psnr_pooled": round(psnr_pooled(pred_rgb, truth_rgb), 3),
        "psnr_boot95": [round(v, 3) for v in bootstrap_ci(per)],
        "chroma_l1": round(cs.chroma_l1(cb_p, cr_p, cb_t, cr_t), 4),
        "colourfulness": round(cs.colourfulness(cb_p, cr_p), 4),
        "colourfulness_truth": round(cs.colourfulness(cb_t, cr_t), 4),
        "hue_err_deg": round(hue_err(cb_p, cr_p, cb_t, cr_t), 1),
        "chroma_radius_mean": round(float(np.hypot(cb_p, cr_p).mean()), 2),
        "clip40_hits": int((per > CLIP_DB).sum()),
    }
    if baseline_per_photo is not None:
        out["beats_baseline"] = int((per > baseline_per_photo).sum())
        out["beats_baseline_pct"] = round(float((per > baseline_per_photo).mean() * 100), 1)
    return out


def hue_err(cb_p, cr_p, cb_t, cr_t, min_radius: float = 20.0) -> float:
    m = np.hypot(cb_t, cr_t) > min_radius
    if not m.any():
        return float("nan")
    dot = np.clip((cb_p[m] * cb_t[m] + cr_p[m] * cr_t[m]) /
                  (np.hypot(cb_p[m], cr_p[m]) * np.hypot(cb_t[m], cr_t[m]) + 1e-9), -1, 1)
    return float(np.degrees(np.arccos(dot)).mean())


def baseline_photos(rgb: np.ndarray) -> np.ndarray:
    """The baseline the task defines: the photo's own luminance, chroma 0 everywhere.

    Not `gray = (R+G+B)/3` - the input *is* Y, and writing Y back through the inverse
    transform is what "no colour" means in this colour space. The residue against the
    photo is float rounding plus the out-of-gamut clip, a few thousandths of 255.
    """
    y, _, _ = cs.ycc(rgb)
    return cs.rgb_of(y, np.zeros_like(y), np.zeros_like(y))


def baseline_per_photo(rgb: np.ndarray) -> np.ndarray:
    """Per-photo PSNR of the gray baseline against itself - `score`'s `baseline_per_photo`."""
    return psnr_photos(baseline_photos(rgb), rgb)


# --------------------------------------------------------------------------- the noise floor --
def bootstrap_ci(values: np.ndarray, draws: int = DRAWS, seed: int = SEED) -> tuple:
    """Percentile bootstrap over photos. `sd` is what a *difference* has to be read against."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    drawn = rng.choice(v, size=(draws, len(v)), replace=True).mean(1)
    return tuple(np.percentile(drawn, [2.5, 97.5]).tolist())


def bootstrap_sd(values: np.ndarray, draws: int = DRAWS, seed: int = SEED) -> float:
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    return float(rng.choice(v, size=(draws, len(v)), replace=True).mean(1).std(ddof=1))


def paired(a_per_photo: np.ndarray, b_per_photo: np.ndarray,
           draws: int = DRAWS, seed: int = SEED) -> dict:
    """The paired difference a - b, per photo. This is the statistic a claim rests on.

    The photo-to-photo spread is 13-17 dB of sd, so two predictors 0.4 dB apart on 100
    photos are indistinguishable *unpaired*; differencing per photo cancels it and the sd
    of the difference is what the 95% interval is built from. Positive = a has the higher
    PSNR. Ties (both photos perfect, both 120 dB) are counted, not discarded.
    """
    d = np.asarray(a_per_photo, np.float64) - np.asarray(b_per_photo, np.float64)
    rng = np.random.default_rng(seed)
    drawn = rng.choice(d, size=(draws, len(d)), replace=True).mean(1)
    return {"n": int(len(d)), "mean": round(float(d.mean()), 3),
            "sd": round(float(d.std(ddof=1)), 3),
            "sem": round(float(d.std(ddof=1) / np.sqrt(len(d))), 3),
            "ci95": [round(v, 3) for v in np.percentile(drawn, [2.5, 97.5])],
            "wins_a": int((d > 0).sum()), "wins_b": int((d < 0).sum()),
            "ties": int((d == 0).sum()),
            "in_sds": round(float(abs(d.mean()) / (d.std(ddof=1) / np.sqrt(len(d)))), 1)}


def quintiles(per_photo: np.ndarray, baseline_per_photo: np.ndarray,
              rgb_truth: np.ndarray, n: int = 5) -> list[dict]:
    """The same comparison split by how colourful the photo already is.

    The audit's section 7: 14% of the photos are nearly colourless, their error is tiny, so
    on a log scale they are worth a huge number of dB and any colour added to them is a pure
    loss. A single mean hides that; these rows are where "the metric rewards refusing to
    colour" is either seen or not.
    """
    _, cb, cr = cs.ycc(rgb_truth)
    content = np.hypot(cb, cr).mean((1, 2))
    edges = np.quantile(content, np.linspace(0, 1, n + 1))
    rows = []
    for k in range(n):
        m = (content >= edges[k]) & (content <= edges[k + 1] if k == n - 1
                                     else content < edges[k + 1])
        if not m.any():
            continue
        d = np.asarray(per_photo)[m] - np.asarray(baseline_per_photo)[m]
        rows.append({"quintile": k + 1, "chroma_range": f"{edges[k]:.1f}-{edges[k + 1]:.1f}",
                     "n": int(m.sum()), "chroma": round(float(content[m].mean()), 1),
                     "psnr": round(float(np.asarray(per_photo)[m].mean()), 2),
                     "psnr_gray": round(float(np.asarray(baseline_per_photo)[m].mean()), 2),
                     "delta": round(float(d.mean()), 2),
                     "beats_gray_pct": round(float((d > 0).mean() * 100), 1)})
    return rows


# ---------------------------------------------------------------------------- the bin route --
def bin_accuracy(pred_index: np.ndarray, truth_index: np.ndarray) -> dict:
    """Top-1 accuracy on the colour bins, and nothing else should be read off it alone.

    Always quote it beside `ceilings()["accuracy_pct"]`, which is what a per-pixel colour
    *lookup table* scores at the same grid - 18.8% at a 32x32 grid, 30.1% at the 16x16 grid
    route B trains on. Accuracy near that number means the network has learned no more about
    colour than a histogram of it; accuracy well above means it has used the full-resolution
    luminance and its neighbourhood, which a histogram cannot see. Accuracy on its own is
    meaningless here: 30% is near the top of what the pixels say, and the pictures it goes
    with are dull, because the top-1 rule answers the modal cell.
    """
    ok = pred_index == truth_index
    return {"accuracy_pct": round(float(ok.mean() * 100), 2),
            "n_pixels": int(ok.size),
            "per_photo_median_pct": round(float(np.median(ok.mean((1, 2))) * 100), 2)}


def binned_truth(rgb: np.ndarray, grid: int) -> np.ndarray:
    """The truth with its chroma snapped to bin centres: the ceiling of the bin route."""
    y, cb, cr = cs.ycc(rgb)
    cb2, cr2 = cs.bin_centre(cs.bin_index(cb, cr, grid), grid)
    return cs.rgb_of(y, cb2, cr2)


def ceilings(train_rgb: np.ndarray, grid: int, yb: int = YB) -> dict:
    """What colour-by-lookup-table can do, measured on the training photos at a given grid.

    Conditions on a `yb`-bin histogram of Y - the same conditioning the audit's lookup tables
    used - and returns, for the *chroma* grid in use:

      `accuracy_pct`  the best per-pixel top-1 accuracy of that lookup table: sum over Y bins
                      of p(Y) * max_k p(bin = k | Y).
      `ce_floor_nats` H(C | Y_64bin), the cross-entropy that table cannot beat, and
                      `entropy_nats` = H(C); their difference is all the *coarsely quantised*
                      luminance says about colour.
      `cells_*`       how populated the grid is: the top cell's share of the training pixels,
                      and how many cells hold < 100 (the imbalance route B's prior-pull is for).

    THESE BOUND THE LOOKUP TABLE, NOT THE NETWORK, and the difference is the interesting
    part of the result. A network sees the luminance at full resolution and in a neighbourhood
    around each pixel; the table sees one 64-bin number. So a trained model can land *below*
    `ce_floor_nats` and above `accuracy_pct` without either number being wrong - route B did,
    2.363 nats against this 2.377 after two epochs on a quarter of the data. Quote them as
    "what a histogram achieves", never as "what no model can beat". The one figure here that
    is a hard ceiling of route B's own decoder is the PSNR of `binned_truth`, which cannot be
    exceeded by any predictor that can only emit bin centres.
    """
    y, cb, cr = cs.ycc(train_rgb)
    yb_idx = np.clip((y / 256 * yb).astype(np.int64), 0, yb - 1)
    cell = cs.bin_index(cb, cr, grid).ravel()
    yb_idx = yb_idx.ravel()
    n_cells = grid * grid
    joint = np.zeros((yb, n_cells), np.float64)
    np.add.at(joint, (yb_idx, cell), 1.0)
    y_count = joint.sum(1)
    p_y = y_count / y_count.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        # denom must stay (yb, 1): `np.where(y_count > 0, y_count[:, None], 1)` looks
        # equivalent and is not - the (yb,) condition broadcasts against the (yb, 1) array
        # to (yb, yb), which then fails to divide the (yb, n_cells) joint table.
        denom = np.where(y_count > 0, y_count, 1.0)[:, None]
        cond = joint / denom
    p_cell = joint.sum(0) / joint.sum()
    ent_cy = float(-(np.where(cond > 0, cond * np.log(cond + 1e-300), 0).sum(1) * p_y).sum())
    ent_c = float(-(np.where(p_cell > 0, p_cell * np.log(p_cell + 1e-300), 0)).sum())
    counts = joint.sum(0)
    return {"grid": int(grid), "yb": int(yb), "n_pixels": int(len(cell)),
            "accuracy_pct": round(float((p_y * np.nan_to_num(cond.max(1))).sum() * 100), 2),
            "ce_floor_nats": round(ent_cy, 3), "entropy_nats": round(ent_c, 3),
            "mutual_info_nats": round(ent_c - ent_cy, 3),
            "perplexity_cy": round(float(np.exp(ent_cy)), 1),
            "cells_populated": int((counts > 0).sum()), "cells_total": int(n_cells),
            "top_cell_share_pct": round(float(counts.max() / counts.sum() * 100), 2),
            "cells_lt_100": int((counts < 100).sum()), "cells_lt_10000": int((counts < 10000).sum())}


# -------------------------------------------------------------------------------- controls --
def control_photos(y: np.ndarray, cb: np.ndarray, cr: np.ndarray, seed: int = SEED) -> dict:
    """Two predictions that must score *worse* than gray, from the model's own chroma.

    A metric that cannot tell a right answer from a plausible-looking wrong one is not
    measuring anything, and on this task the wrong answers are as colourful as the right
    ones - colourfulness alone cannot separate them, so these do:

      `shuffled`  photo i gets photo i+1's predicted chroma: the right *kind* of colour in
                  the wrong place. Same colourfulness as the model, PSNR that must fall
                  below gray's, because the only thing keeping PSNR up is agreement with Y.
      `rolled`    the chroma rolled by half a photo: the same array, still coherent inside
                  itself, aligned with nothing.
    """
    n = len(cb)
    shift = np.roll(np.arange(n), 1)                    # photo i gets photo i+1's chroma
    if n > 1:
        shift = np.roll(shift, n // 2)                  # so photo 0 is not given its own chroma
    assert not (shift == np.arange(n)).any(), "a derangement is required, or a control is the answer"
    return {"shuffled": cs.rgb_of(y, cb[shift], cr[shift]),
            "rolled": cs.rgb_of(y, np.roll(cb, 32, (1, 2)), np.roll(cr, 32, (1, 2)))}


# ------------------------------------------------------------------------------- printing --
def summarize(name: str, s: dict) -> str:
    """One line per predictor, the same columns in every log."""
    lo, hi = s["psnr_boot95"]
    beat = (f"  beats gray {s['beats_baseline']:>3}/{s['n']}" if "beats_baseline" in s else "")
    return (f"  {name:<22} PSNR {s['psnr_mean']:>6.2f} dB [{lo:>6.2f}, {hi:>6.2f}]  "
            f"med {s['psnr_median']:>6.2f}  clip40 {s['psnr_clip40']:>6.2f}  "
            f"pooled {s['psnr_pooled']:>6.2f}  chromaL1 {s['chroma_l1']:.4f}  "
            f"colourful {s['colourfulness']:.3f}  hue {s['hue_err_deg']:>5.1f}{beat}")


def table(rows: list[tuple[str, dict]], title: str = "") -> str:
    lines = ([title] if title else [])
    lines += [summarize(n, s) for n, s in rows]
    return "\n".join(lines)


if __name__ == "__main__":
    # The metric against its own definitions, on synthetic photos with a known answer.
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 255, (4, 64, 64)).astype(np.float32)
    cb = rng.uniform(-40, 40, (4, 64, 64)).astype(np.float32)
    cr = rng.uniform(-40, 40, (4, 64, 64)).astype(np.float32)
    photo = cs.rgb_of(y, cb, cr)

    print("a perfect prediction scores:")
    print(summarize("truth", score(photo, photo)))
    print("  psnr_mean is the ceiling this file can report:",
          f"{score(photo, photo)['psnr_mean']:.1f} dB = 10*log10(1/EPS), i.e. EPS={EPS:g}; "
          "that is the epsilon, not the model")
    print("\nthe gray baseline, same function:")
    gray = baseline_photos(photo)
    print(summarize("gray", score(gray, photo, baseline_per_photo=baseline_per_photo(photo))))
    s = score(gray, photo)
    print(f"  chromaL1 {s['chroma_l1']:.4f} == 2 x truth colourfulness "
          f"{s['colourfulness_truth']:.4f} -> "
          f"{abs(s['chroma_l1'] - 2 * s['colourfulness_truth']) < 2e-4}")
    print(f"  colourfulness {s['colourfulness']:.4f} (must be 0.0000: gray has no chroma)")
    print("\nthe controls, which must be worse than gray:")
    for name, pred in control_photos(*cs.ycc(photo)).items():
        print(summarize(name, score(pred, photo, baseline_per_photo=baseline_per_photo(photo))))
