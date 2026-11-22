"""The two constants this task needs, chosen on the 100 validation photos. `python calibrate.py`.

Route A predicts chroma and needs one number, `alpha`, to say how much of it to keep: an L1
fitting chroma regression returns a shrunk answer, and the audit measured that shrunk answer
as keeping 29% of the photos' colour - which is the "dull brownish gray" the task warns about.
Route B needs `gamma`, the strength of the prior pull at inference: `argmax p` collapses onto
the modal bin, 26.7% of all training pixels, and the pull is what pushes pixels off it.

THE RULES, written before either trained model was decoded:

  CA  One rule for both constants, so the two routes are chosen the same way: maximise the
      per-photo mean PSNR subject to colourfulness >= FLOOR * the photos' own, FLOOR = 0.5.
  CB  The floor is not decoration. Without it the rule is degenerate on this data: per-photo
      mean PSNR is maximised by predicting no colour at all (alpha = 0 for route A; for route
      B, answering the modal bin everywhere), because gray already scores 26.94 dB and any
      wrong colour costs more than it earns. The audit measured that maximum at alpha = 0
      before any network existed. So "best PSNR" alone means "no picture", and the question
      worth answering is how much PSNR survives once the model is required to be colourful.
  CC  The whole curve is reported, along with what five alternative rules would have picked -
      unconstrained per-photo PSNR, pooled PSNR, PSNR clipped at 40 dB, minimum chroma L1, and
      minimum hue error - each in the "alternatives" block of results/calibration.json, with the
      whole 41-point alpha grid and 21-point gamma grid beside it and the curves in
      figures/calibration.png. The chosen value is not offered as the only defensible one; the
      reader can see what the others cost. (On this data the six answers differ, and two of them
      are argmaxes over a curve that is flat to within 0.2 degrees - which is itself the point of
      reporting them.)
  CD  Neither constant is chosen by looking at a picture. The photo grid in
      `evaluate_task6.py` (seven rows: two extremes, one already-black-and-white, four from a
      seeded draw) is chosen by a rule over the *test* photos afterwards and is
      presented as illustration, never as evidence.
  CE  Nothing here reads a test photo: alpha and gamma are decided on the validation split,
      and the test photos are scored once, afterwards, by `evaluate_task6.py` (its ET1).
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

import colorspace as cs
import col6_data as d6
import figs
import fitting
import metrics as M
import spec

OUT = d6.HERE / "results"
FLOOR = 0.5                 # CA: colourfulness must be at least half the photos' own
ALPHAS = np.round(np.arange(0.0, 2.001, 0.05), 3)
GAMMAS = np.round(np.arange(0.0, 1.001, 0.05), 3)


def pick(rows: list[dict], key: str, best: str = "max") -> float:
    v = [r[key] for r in rows]
    i = int(np.argmax(v) if best == "max" else np.argmin(v))
    return rows[i]["x"]


def apply_floor(rows: list[dict], floor: float) -> float | None:
    ok = [r for r in rows if r["colourfulness"] >= floor * r["colourfulness_truth"]]
    if not ok:
        return None
    return ok[int(np.argmax([r["psnr_mean"] for r in ok]))]["x"]


def row(x: float, rgb: np.ndarray, truth: np.ndarray, base: np.ndarray, extra: dict) -> dict:
    s = M.score(rgb, truth, baseline_per_photo=base)
    return {"x": round(float(x), 3), **{k: s[k] for k in
            ("psnr_mean", "psnr_clip40", "psnr_pooled", "chroma_l1", "colourfulness",
             "colourfulness_truth", "hue_err_deg", "beats_baseline_pct")}, **extra}


def table(rows: list[dict], name: str, chosen: float, step: float = 0.1) -> None:
    print(f"\n  {name:>6} {'PSNR':>7} {'clip40':>7} {'pooled':>7} {'chromaL1':>9} "
          f"{'colourful':>10} {'hue err':>8} {'beats gray':>11}")
    for r in rows:
        if abs(round(r[name] / step) - r[name] / step) > 1e-9:
            continue
        star = "  <- chosen" if abs(r[name] - chosen) < 1e-9 else ""
        print(f"  {r[name]:>6.2f} {r['psnr_mean']:>7.2f} {r['psnr_clip40']:>7.2f} "
              f"{r['psnr_pooled']:>7.2f} {r['chroma_l1']:>9.4f} {r['colourfulness']:>10.4f} "
              f"{r['hue_err_deg']:>8.2f} {r['beats_baseline_pct']:>10.1f}%{star}")


def alternatives(rows: list[dict], chosen: float) -> dict:
    return {
        "chosen (CA: PSNR subject to the colourfulness floor)": chosen,
        "unconstrained per-photo PSNR": pick(rows, "psnr_mean"),
        "pooled PSNR": pick(rows, "psnr_pooled"),
        "clipped-at-40 dB PSNR": pick(rows, "psnr_clip40"),
        "minimum chroma L1": pick(rows, "chroma_l1", "min"),
        "maximum hue accuracy (min hue error)": pick(rows, "hue_err_deg", "min"),
    }


def figure(panels: list[tuple[str, str, list[dict], float]]) -> None:
    figs.style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(panels), 3, figsize=(9.2, 2.6 * len(panels)))
    axes = np.atleast_2d(axes)
    for i, (label, xname, rows, chosen) in enumerate(panels):
        x = [r[xname] for r in rows]
        for j, (key, unit, colour) in enumerate([
                ("psnr_mean", "dB, per photo", figs.METHOD_COLORS["A"]),
                ("colourfulness", "|chroma| / 255", figs.METHOD_COLORS["B+"]),
                ("chroma_l1", "mean |dCb| + |dCr| / 255", figs.METHOD_COLORS["B"])]):
            ax = axes[i, j]
            ax.plot(x, [r[key] for r in rows], color=colour, lw=1.6,
                    label=f"{label}: {key}")
            if key == "psnr_mean":
                ax.plot(x, [r["psnr_clip40"] for r in rows], color=colour, lw=1.2, ls="--",
                        label="clipped at 40 dB")
                ax.plot(x, [r["psnr_pooled"] for r in rows], color=colour, lw=1.2, ls=":",
                        label="pooled")
            if key == "colourfulness":
                ax.axhline(rows[0]["colourfulness_truth"], color=figs.GREY, ls="--", lw=1.0,
                           label=f"the photos' own {rows[0]['colourfulness_truth']:.4f}")
                ax.axhline(FLOOR * rows[0]["colourfulness_truth"], color=figs.RED, ls="-.",
                           lw=1.0, label=f"the floor, {FLOOR:.1f} x")
                ax.axhline(0.0, color=figs.GREY, lw=0.8, label="gray")
            ax.axvline(chosen, color=figs.INK, lw=1.0, ls=":",
                       label=f"chosen {label} = {chosen}")
            ax.set_xlabel(f"{label} (dotted line: chosen)", fontsize=7)
            ax.set_ylabel(unit, fontsize=7)
            ax.tick_params(labelsize=7)
            figs.legend_on(ax, loc="best")
            if j:
                ax.set_yticklabels([])
    fig.suptitle("Every constant chosen on the validation split: PSNR alone would pick no "
                 "colour at all, so colourfulness >= half the photos' own is required first",
                 fontsize=8.5)
    figs.save(fig, "calibration.png")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    fitting.threads(6)
    t0 = time.perf_counter()
    print(spec.text())

    data = d6.build(verbose=False)
    valid = d6.arrays(data, "valid")
    base = M.baseline_per_photo(valid)
    truth_cf = cs.colourfulness(*cs.ycc(valid)[1:])
    prior = fitting.prior(d6.arrays(data, "train"), cs.GRID)
    ceil16 = M.ceilings(d6.arrays(data, "train"), cs.GRID)
    print(f"\ncalibrating on the {len(valid)} validation photos; their own colourfulness is "
          f"{truth_cf:.4f}, the floor is {FLOOR:.1f} x that = {FLOOR * truth_cf:.4f}; "
          f"gray scores {base.mean():.2f} dB")

    out = {"floor": FLOOR, "n_valid": int(len(valid)), "gray_psnr_mean": round(float(base.mean()), 3),
           "colourfulness_truth": round(truth_cf, 5), "thread_note": "6 threads"}

    # ---- route A: one scalar --------------------------------------------------------------- #
    model_a, meta_a = fitting.load_model(OUT / "model_A.pt")
    one = fitting.decode(model_a, valid, alpha=1.0)
    y = cs.ycc(valid)[0]
    rows = []
    for a in ALPHAS:                                   # scaling chroma is exact, so decode once
        rows.append(row(a, cs.rgb_of(y, one["cb"] * a, one["cr"] * a), valid, base, {}))
    alpha = apply_floor(rows, FLOOR)
    assert alpha is not None, "no alpha clears the colourfulness floor - report that, do not lower it"
    print(f"\nROUTE A - alpha over {len(ALPHAS)} values, "
          f"chosen {alpha} by rule CA (PSNR subject to the floor)")
    table(rows, "x", alpha)
    out["alpha"] = {"grid": [float(v) for v in ALPHAS], "rows": rows, "chosen": alpha,
                    "rule": "CA: maximise per-photo mean PSNR subject to colourfulness >= "
                            f"{FLOOR} x the photos' own",
                    "alternatives": alternatives(rows, alpha),
                    "gray_psnr": round(float(base.mean()), 3),
                    "uncalibrated_psnr": rows[int(np.where(ALPHAS == 1.0)[0][0])]["psnr_mean"]}

    # ---- route B: the prior pull ----------------------------------------------------------- #
    model_b, meta_b = fitting.load_model(OUT / "model_B.pt")
    # The prior the weights were trained with, off the checkpoint, not recomputed: a mismatch
    # here would mean gamma is pulling towards a distribution the model never saw.
    assert np.abs(meta_b["prior"] - prior).max() < 1e-9, "checkpoint prior != train prior"
    rows_b = []
    for g in GAMMAS:
        d = fitting.decode(model_b, valid, prior, gamma=float(g))
        acc = M.bin_accuracy(d["bin"], cs.bin_index(*cs.ycc(valid)[1:], cs.GRID))
        rows_b.append(row(g, d["rgb"], valid, base, {"accuracy_pct": acc["accuracy_pct"],
                                                     "accuracy_ceiling_pct": ceil16["accuracy_pct"]}))
    gamma = apply_floor(rows_b, FLOOR)
    assert gamma is not None, "no gamma clears the floor"
    print(f"\nROUTE B - gamma over {len(GAMMAS)} values, chosen {gamma} by rule CA")
    print(f"  the accuracy ceiling at this grid is {ceil16['accuracy_pct']}% - accuracy is "
          f"reported against it and never alone")
    print(f"\n  {'gamma':>6} {'PSNR':>7} {'clip40':>7} {'pooled':>7} {'chromaL1':>9} "
          f"{'colourful':>10} {'hue err':>8} {'beats gray':>11} {'bin acc':>9}")
    for r in rows_b:
        if abs(round(r["x"] / 0.1) - r["x"] / 0.1) > 1e-9:
            continue
        star = "  <- chosen" if abs(r["x"] - gamma) < 1e-9 else ""
        print(f"  {r['x']:>6.2f} {r['psnr_mean']:>7.2f} {r['psnr_clip40']:>7.2f} "
              f"{r['psnr_pooled']:>7.2f} {r['chroma_l1']:>9.4f} {r['colourfulness']:>10.4f} "
              f"{r['hue_err_deg']:>8.2f} {r['beats_baseline_pct']:>10.1f}% "
              f"{r['accuracy_pct']:>8.2f}%{star}")
    out["gamma"] = {"grid": [float(v) for v in GAMMAS], "rows": rows_b, "chosen": gamma,
                    "rule": "CA, same as route A's alpha",
                    "accuracy_ceiling_pct": ceil16["accuracy_pct"],
                    "alternatives": alternatives(rows_b, gamma)}

    figure([("alpha", "x", rows, alpha), ("gamma", "x", rows_b, gamma)])
    (OUT / "calibration.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    a_row = next(r for r in rows if r["x"] == alpha)
    g_row = next(r for r in rows_b if r["x"] == gamma)
    print(f"\nCHOSEN: alpha = {alpha} gives {a_row['psnr_mean']:.2f} dB at colourfulness "
          f"{a_row['colourfulness']:.4f}; gamma = {gamma} gives {g_row['psnr_mean']:.2f} dB at "
          f"{g_row['colourfulness']:.4f}; gray is {base.mean():.2f} dB at colourfulness 0.")
    for name, val in out["alpha"]["alternatives"].items():
        r = next(r for r in rows if r["x"] == val)
        print(f"  route A, {name:<52} alpha {val:<6} {r['psnr_mean']:.2f} dB, colourful "
              f"{r['colourfulness']:.4f}")
    for name, val in out["gamma"]["alternatives"].items():
        r = next(r for r in rows_b if r["x"] == val)
        print(f"  route B, {name:<52} gamma {val:<6} {r['psnr_mean']:.2f} dB, colourful "
              f"{r['colourfulness']:.4f}")
    print(f"\nwrote results/calibration.json ({(time.perf_counter() - t0) / 60:.1f} min)")
    print("next: python evaluate_task6.py --valid   (a dry run on valid, then the real thing)")


if __name__ == "__main__":
    main()
