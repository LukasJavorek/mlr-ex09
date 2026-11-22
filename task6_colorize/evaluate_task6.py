"""The one look at the 500 test photos. `python evaluate_task6.py`: 2.5 min measured, of
which about 2 is the 3325-photo valrest confirmation; `--valid --no-valrest` takes 0.2 min.

    python evaluate_task6.py             the real thing: the test split
    python evaluate_task6.py --valid     the same code on the validation split, as many times
                                         as you like - it is a dry run, not a second opinion
    python evaluate_task6.py --valid --valrest
                                         the dry run with the 3325-photo confirmation too, which
                                         is what the test run does; without this flag the --valid
                                         path never executes that block, and it would then be the
                                         only untested code standing between the test split and
                                         its results file

THE RULES THIS SCRIPT IS BUILT AROUND.

  ET1  The test photos are *scored* here and nowhere else in the folder, and they decide nothing.
       `col6_data.py` builds the array and `smoke_test.py` checks its shape and sha1 against the
       audit's - checks on bytes that were fixed before this task existed, on no model and on no
       number any model produces - but no other script decodes, scores, plots or reads a quantity
       out of the test pixels. Every constant, width, epoch, learning rate, batch size, pixel
       count, alpha and gamma was chosen on the 100 validation photos by `validate_choices.py`
       and `calibrate.py`, before this file ran. Nothing here can feed back: no model is fitted,
       and no threshold anywhere below selects anything.
  ET2  Every predictor goes through the same `metrics.score` call on the same photos with the
       same gray baseline - including gray, which is re-measured here rather than quoted from
       the audit, so the baseline in the table is produced by the code being judged.
  ET3  The two controls ride along: the model's own chroma on the wrong photo, and rolled by
       half a photo. On the test photos too, because a metric that only behaves on the split
       it was written against is not a metric.
  ET4  `valrest` - the 3325 validation photos the handout leaves unused - is reported as a
       confirmation column. Nothing was chosen on it either, so agreement with the 100-photo
       numbers means the 100 were not a fluke; disagreement would mean the choices are shaky,
       and it would be reported that way.
  ET5  The figure's photos are chosen by rule (below), printed, and reproducible. They
       illustrate the numbers; they are not evidence for them, and the caption says so.
"""
from __future__ import annotations

import argparse
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
METHODS = ("gray", "A", "B", "B+", "binned truth")


def rule_photos(rgb: np.ndarray, n_random: int = 4, seed: int = 0) -> tuple[list[int], list[str]]:
    """ET5: the photos of the figure, chosen by rule rather than by eye.

    The most colourful and the least colourful test photo are where any colouriser looks best
    and worst, so showing only those would be cherry-picking in both directions. The fix is to
    name them as the extremes in the caption and add a photo that is already black and white
    (the case where a method cannot add colour without inventing it), plus `n_random` photos
    from a seeded draw that nobody got to choose: 3 + n_random rows, seven in the shipped
    figure. The count is not restated in prose anywhere else - an earlier draft of this line
    said "eight" after the last arm of the rule lost one, and nothing failed.
    """
    cf = cs.colourfulness_photos(*cs.ycc(rgb)[1:])
    order = np.argsort(cf)
    idx = [int(np.argmax(cf)), int(order[3])]              # brightest, and a very dull one
    grey = np.flatnonzero(d6.exactly_grey(rgb))
    idx.append(int(grey[0]) if len(grey) else int(np.argsort(cf)[1]))
    labels = ["most colourful", "among the dullest",
              "already black and white" if len(grey) else "second dullest"]
    pool = np.setdiff1d(np.arange(len(rgb)), idx)
    idx += [int(i) for i in np.random.default_rng(seed).choice(pool, n_random, replace=False)]
    labels += [f"seeded draw {k + 1}" for k in range(n_random)]
    return idx, labels


def predict_all(models: dict, rgb: np.ndarray, cal: dict, prior) -> dict[str, dict]:
    """Every method, decoded once each: its photos, its chroma and (route B) its bin indices."""
    out = {"gray": {"rgb": M.baseline_photos(rgb)}}
    if "A" in models:
        out["A"] = fitting.decode(models["A"], rgb, alpha=cal["alpha"]["chosen"])
    if "B" in models:
        out["B"] = fitting.decode(models["B"], rgb, prior, gamma=0.0)
        out["B+"] = fitting.decode(models["B"], rgb, prior, gamma=cal["gamma"]["chosen"])
    out["binned truth"] = {"rgb": M.binned_truth(rgb, cs.GRID)}   # route B's hard ceiling
    return {k: v for k, v in out.items() if k in METHODS}


def photo_figure(dec: dict, truth: np.ndarray, idx: list[int], labels: list[str],
                 tag: str) -> None:
    figs.style()
    import matplotlib.pyplot as plt
    cols = [c for c in ("truth", "gray", "A", "B", "B+") if c == "truth" or c in dec]
    names = {"truth": "the photo", "gray": "gray (the baseline)",
             "A": "A, alpha calibrated", "B": "B, argmax p", "B+": "B+, prior pull"}
    fig, axes = plt.subplots(len(idx), len(cols), figsize=(2.0 * len(cols), 2.05 * len(idx)),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    for r, (i, lab) in enumerate(zip(idx, labels)):
        for c, col in enumerate(cols):
            img = truth[i] if col == "truth" else dec[col]["rgb"][i]
            figs.photo_box(axes[r, c], img)
            if r == 0:
                axes[r, c].set_title(names[col], fontsize=7.5, loc="left")
        axes[r, 0].set_ylabel(lab, fontsize=6.5, rotation=0, ha="right", va="center",
                              labelpad=6)
    fig.suptitle(f"{tag}: photos chosen by rule ET5 (the extremes, an already-grey one, and a "
                 f"seeded draw) - these illustrate the table, they are not the evidence",
                 fontsize=8.5)
    figs.save(fig, f"photos_{tag.replace(' ', '_')}.png")


def chroma_figure(dec: dict, truth: np.ndarray, tag: str) -> None:
    """Where each method's chroma actually lands, as a 2D histogram over the Cb-Cr plane.

    One shared colour scale across panels, on a log axis, one measure per panel: the panels
    differ only in which method produced the chroma, so a bright blob near the origin in the B
    panel and a spread one in the B+ panel is the prior pull, seen directly.

    Two choices worth stating, because both are the kind of thing that makes a figure lie.
    (1) Gray is a delta function - every one of its
    pixels sits in the single cell at the origin, fraction 1.0 - so including it in the shared
    maximum crushes every other panel into invisibility; it is therefore drawn as a marker at
    the origin and left off the colour scale, which is what it is: a value off the scale.
    (2) On a linear fraction scale the modal cell is two orders of magnitude above the tail
    anyone is interested in, so the scale is logarithmic and shared, and the colourbar carries
    the units.
    """
    figs.style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    cols = [c for c in ("truth", "gray", "A", "B", "B+") if c == "truth" or c in dec]
    fig, axes = plt.subplots(1, len(cols), figsize=(2.3 * len(cols), 2.6), squeeze=False)
    lim, hmax = 128.0, 0.0
    hists = {}
    for col in cols:
        img = truth if col == "truth" else dec[col]["rgb"]
        _, cb, cr = cs.ycc(img)
        h, _, _ = np.histogram2d(cb.ravel(), cr.ravel(), bins=cs.GRID,
                                 range=[[-lim, lim], [-lim, lim]])
        hists[col] = h / max(h.sum(), 1)
        if col != "gray":
            hmax = max(hmax, hists[col].max())
    image = None
    for ax, col in zip(axes[0], cols):
        h = hists[col]
        if col == "gray":                     # off the scale by construction: one cell, 100%
            # an all-masked image with the same extent: `aspect="equal"` shrinks the other
            # panels' boxes to square, and without this the gray panel keeps the full-height
            # box, which lifts its title up into the figure title
            ax.imshow(np.ma.masked_all((2, 2)), extent=[-lim, lim, -lim, lim], aspect="equal")
            ax.scatter([0], [0], s=46, marker="s", color=figs.INK, zorder=3)
        else:
            image = ax.imshow(h.T, origin="lower", extent=[-lim, lim, -lim, lim],
                              aspect="equal", cmap="Greys",
                              norm=LogNorm(vmin=hmax * 1e-4, vmax=hmax), interpolation="nearest")
            if col in ("B", "B+"):            # cell edges = the grid the argmax chooses between
                step = 2 * lim / cs.GRID
                for k in np.arange(-lim, lim + 1, step):
                    ax.axhline(k, color=figs.GREY, lw=0.25, alpha=0.45)
                    ax.axvline(k, color=figs.GREY, lw=0.25, alpha=0.45)
        ax.axhline(0, color=figs.GREY, lw=0.6)
        ax.axvline(0, color=figs.GREY, lw=0.6)
        note = {"gray": "\nchroma 0: one cell, off this scale",
                "B": "\nthe 16x16 grid overlaid", "B+": "\nthe 16x16 grid overlaid"}.get(col, "")
        ax.set_title(col + note, fontsize=7, loc="left")
        ax.set_xlabel("Cb", fontsize=7)
        ax.set_ylabel("Cr" if col == "truth" else "", fontsize=7)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.tick_params(labelsize=6.5)
    ticks = [hmax * 1e-4, hmax * 1e-2, hmax]      # four decades, labelled from the data itself
    bar = fig.colorbar(image, ax=axes[0].tolist(), fraction=0.026, pad=0.02, ticks=ticks)
    bar.ax.set_yticklabels([f"{v * 100:.3g}%" for v in ticks], fontsize=6.5)
    bar.minorticks_off()
    bar.set_label("fraction of all pixels in the cell (log)", fontsize=7)
    bar.outline.set_visible(False)
    fig.suptitle(f"predicted chroma over the {tag} photos: one shared log scale, so the tail "
                 f"and the mode are visible together (gray is off that scale, by construction)",
                 fontsize=8.5)
    figs.save(fig, f"chroma_support_{tag.replace(' ', '_')}.png")


def quintile_rows(per_photo: dict, base: np.ndarray, rgb: np.ndarray) -> dict:
    """PSNR change against gray, by how colourful the photo already was. Data, no figure."""
    names = [n for n in per_photo if n != "gray"]
    return {n: M.quintiles(per_photo[n], base, rgb) for n in names}


def quintile_figure(rows: dict, tag: str) -> None:
    """The rows above as bars. Kept apart from `quintile_rows` so the numbers are on disk
    before any drawing code runs - a figure that crashes must not take a results file with it."""
    figs.style()
    import matplotlib.pyplot as plt
    names = list(rows)
    fig, ax = plt.subplots(figsize=(5.6, 2.9), constrained_layout=True)
    width = 0.8 / max(len(names), 1)
    for k, name in enumerate(names):
        d = [r["delta"] for r in rows[name]]
        bars = ax.bar(np.arange(len(d)) + k * width - 0.4 + width / 2, d, width * 0.9,
                      color=figs.METHOD_COLORS.get(name, figs.SERIES[k % 4]), label=name)
        if name == "binned truth":       # a ceiling, not a method: hatched so it reads as one
            for b in bars:
                b.set_hatch("//")
                b.set_edgecolor(figs.SURFACE)
                b.set_linewidth(0.4)
    ax.axhline(0, color=figs.INK, lw=0.9)
    # A symmetric log below +-2 dB: the dullest quintile runs to -20 dB, the rest live inside
    # +-2, and a linear axis would spend all its height on the one bar everyone already knows
    # about. The tick labels say the real values, so nothing is hidden by the compression.
    ax.set_yscale("symlog", linthresh=2.0)
    ticks = [-20, -10, -5, -2, 0, 2, 5, 10, 20]
    ax.set_yticks(ticks, [f"{t:+d}" if t else "0" for t in ticks])   # plain dB, not 2x10^1
    ax.minorticks_off()
    first = rows[names[0]]
    ax.set_xticks(np.arange(len(first)),
                  [f"Q{r['quintile']}\nchroma {r['chroma']:.1f}\n{r['n']} photos" for r in first],
                  fontsize=6.2)
    ax.set_ylabel("PSNR change against gray, dB\n(log beyond +-2 dB)", fontsize=7.5)
    ax.set_xlabel("the photo's own chroma, dullest quintile on the left", fontsize=7.5)
    ax.tick_params(axis="y", labelsize=6.5)
    figs.legend_on(ax, loc="upper left")
    ax.set_title(f"{tag}: where each method helps and where it hurts, by how colourful the photo "
                 "was already (hatched bars are route B's ceiling, not a method)", fontsize=8.0,
                 loc="left")
    figs.save(fig, f"quintiles_{tag.replace(' ', '_')}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", action="store_true", help="dry run on the validation split")
    ap.add_argument("--no-valrest", action="store_true", help="skip the 3325-photo confirmation")
    ap.add_argument("--valrest", action="store_true",
                    help="also score the 3325-photo confirmation on a --valid dry run, so the "
                         "code path the test run takes is not being executed for the first time "
                         "there (implied when the split is the test one)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    fitting.threads(6)
    t_start = time.perf_counter()

    print(spec.text())
    cal = json.loads((OUT / "calibration.json").read_text(encoding="utf-8"))
    print(f"\nalpha = {cal['alpha']['chosen']} and gamma = {cal['gamma']['chosen']}, both chosen "
          f"on the validation split by rule CA; epochs, widths, learning rates and batch sizes "
          f"came from results/choices.json. Nothing below can change any of them.")

    data = d6.build(verbose=False)
    split = "valid" if a.valid else "test"
    tag = "validation dry run" if a.valid else "test"
    rgb = d6.arrays(data, split)
    prior = fitting.prior(d6.arrays(data, "train"), cs.GRID)
    models = {}
    for route in ("A", "B"):
        path = OUT / f"model_{route}.pt"
        if path.exists():
            models[route], _ = fitting.load_model(path)
        else:
            print(f"  ! {path.name} is missing - the baseline and the ceiling will still score")
    print(f"\nscoring the {split} split: {len(rgb)} photos. "
          + ("DRY RUN on validation photos - the test split is still untouched."
             if a.valid else
             "THIS IS THE ONE LOOK AT THE TEST SPLIT: it happens once and nothing downstream "
             "reads it."))

    dec = predict_all(models, rgb, cal, prior)
    base = M.baseline_per_photo(rgb)
    per_photo = {name: M.psnr_photos(d["rgb"], rgb) for name, d in dec.items()}
    table = {name: M.score(d["rgb"], rgb, baseline_per_photo=base) for name, d in dec.items()}

    # bin accuracy for the two routes that predict bins, from the decodes already done
    truth_idx = cs.bin_index(*cs.ycc(rgb)[1:], cs.GRID)
    ceil16 = M.ceilings(d6.arrays(data, "train"), cs.GRID)
    for name in ("B", "B+"):
        if name in dec:
            table[name].update(M.bin_accuracy(dec[name]["bin"], truth_idx))
            table[name]["lookup_ceiling_pct"] = ceil16["accuracy_pct"]

    keys = [("psnr_mean", "PSNR mean", 2), ("psnr_median", "median", 2),
            ("psnr_clip40", "clip40", 2), ("psnr_pooled", "pooled", 2),
            ("chroma_l1", "chroma L1", 4), ("colourfulness", "colourful", 4),
            ("hue_err_deg", "hue err", 2), ("beats_baseline_pct", "beats gray %", 1)]
    print(f"\n{'method':<14}" + "".join(f"{label:>13}" for _, label, _ in keys)
          + f"{'bin acc':>9}{'/lookup':>9}")
    for name, s in table.items():
        acc = (f"{s.get('accuracy_pct', float('nan')):>9.2f}{ceil16['accuracy_pct']:>9.1f}"
               if name in ("B", "B+") else f"{'-':>9}{'-':>9}")
        print(f"{name:<14}" + "".join(f"{s[k]:>13.{d}f}" for k, _, d in keys) + acc)
    print(f"\n  gray is the baseline the task defines and it is measured here, not quoted: "
          f"{table['gray']['psnr_mean']:.2f} dB mean on these {len(rgb)} photos "
          f"(the audit's number for this split was "
          f"{'26.94' if split == 'valid' else '25.36'} dB).")
    print(f"  bin accuracy is printed beside what a colour lookup table scores at this grid "
          f"({ceil16['accuracy_pct']}%), never on its own.")
    if "B+" in table and float(cal["gamma"]["chosen"]) == 0.0:
        print("  gamma calibrated to 0, so the B+ row is the B row: argmax p already clears "
              "rule CA's colourfulness floor, and every gamma > 0 spends PSNR that nothing "
              "asked for. That is rule CA's measured outcome, not an arm that was skipped - "
              "the whole gamma curve is in results/calibration.json and "
              "figures/calibration.png, and figures/chroma_support_*.png has B and B+ as the "
              "same blob, which is what gamma = 0 means.")

    paired = {}
    for name, p in per_photo.items():
        if name == "gray":
            continue
        q = M.paired(p, per_photo["gray"])
        paired[name] = q
        print(f"  {name:<12} against gray: {q['mean']:+.2f} dB per photo (sd {q['sd']:.2f}, "
              f"95% CI {q['ci95'][0]:+.2f} to {q['ci95'][1]:+.2f}, {q['in_sds']:.1f} sems), "
              f"wins {q['wins_a']}/{q['n']}, ties {q['ties']}")

    print()
    if "B+" in dec:
        y, cb, cr = cs.ycc(dec["B+"]["rgb"])
        for cname, pred in M.control_photos(y, cb, cr).items():
            b2 = M.baseline_per_photo(rgb)
            s = M.score(pred, rgb, baseline_per_photo=b2)
            p = M.paired(M.psnr_photos(pred, rgb), per_photo["gray"])
            print(f"  control {cname:<9} {s['psnr_mean']:.2f} dB ({p['mean']:+.2f} against gray, "
                  f"95% CI {p['ci95'][0]:+.2f} to {p['ci95'][1]:+.2f}), colourfulness "
                  f"{s['colourfulness']:.4f} - the model's own chroma, in the wrong place")
            table[f"control_{cname}"] = s

    confirm = {}
    if not a.no_valrest and (a.valrest or not a.valid):
        vr = d6.arrays(data, "valrest")
        print(f"\nconfirmation on valrest ({len(vr)} photos that decided nothing):")
        vdec = predict_all(models, vr, cal, prior)
        # only the photos are scored here; the chroma and bin indices of 3325 photos would be
        # ~0.4 GB of nothing, and the figures that need them use the test-split decodes instead
        for d in vdec.values():
            for k in ("cb", "cr", "bin"):
                d.pop(k, None)
        vbase = M.baseline_per_photo(vr)
        for name, d in vdec.items():
            s = M.score(d["rgb"], vr, baseline_per_photo=vbase)
            confirm[name] = s
            print(f"  {name:<14} {s['psnr_mean']:7.2f} dB  median {s['psnr_median']:6.2f}  "
                  f"clip40 {s['psnr_clip40']:6.2f}  colourful {s['colourfulness']:.4f}  "
                  f"beats gray {s['beats_baseline_pct']:5.1f}%")

    idx, labels = rule_photos(rgb)
    cf = cs.colourfulness_photos(*cs.ycc(rgb)[1:])
    quint = quintile_rows(per_photo, base, rgb)
    payload = {"split": split, "n_photos": int(len(rgb)), "alpha": cal["alpha"]["chosen"],
               "gamma": cal["gamma"]["chosen"], "table": table, "paired_vs_gray": paired,
               "quintiles": quint, "lookup_accuracy_ceiling_pct": ceil16["accuracy_pct"],
               "lookup_ceiling_note": "this bounds a colour lookup table, not a network - see "
                                      "metrics.ceilings",
               "figure_photos": [{"index": int(i), "why": l,
                                  "colourfulness": round(float(cf[i]), 5)}
                                 for i, l in zip(idx, labels)],
               "valrest_confirmation": confirm,
               "seconds": round(time.perf_counter() - t_start, 1), "threads": 6}
    out_name = "metrics_valid_dryrun.json" if a.valid else "metrics_test.json"
    (OUT / out_name).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote results/{out_name} in {(time.perf_counter() - t_start) / 60:.1f} min")

    # Figures last, on purpose: the numbers above are on disk, so a plotting failure costs a
    # figure and not a results file. All three are drawn in smoke_test.py check 17 first, on
    # synthetic arrays, precisely so that they are not being executed for the first time here.
    print("\nfigure photos (rule ET5): "
          + ", ".join(f"{i} ({l}, colourful {cf[i]:.4f})" for i, l in zip(idx, labels)))
    photo_figure(dec, rgb, idx, labels, tag)
    chroma_figure(dec, rgb, tag)
    quintile_figure(quint, tag)
    print("\nWHAT SHOULD CONVINCE US THERE ARE NO MISTAKES, and where each piece lives:")
    import smoke_test as S      # for the count only: quoting it beats remembering it
    print(f"  * smoke_test.py's {S.CHECKS} checks, run before anything was trained - the metric "
          "reproducing the audit's gray baseline to 0.02 dB, the zero-initialised head scoring "
          "that baseline exactly, the sampled-pixel head being the same function as the full "
          "one, the bin grid reproducing the audit's 31.54 dB ceiling, each control losing to "
          "gray by a paired deficit whose 95% interval excludes zero, and every call site in "
          "this folder bound against the real signature of the function it names;")
    print("  * every choice measured on the 100 validation photos under rules R1-R5 and CA-CE, "
          "written down before the arms ran, with stage 3's seed spread as the yardstick for "
          "which differences mean anything;")
    print("  * the two PSNR conventions printed beside each other everywhere, because on this "
          "data they disagree about which predictor is better and one number alone would have "
          "hidden that;")
    print("  * the valrest column: 3325 photos that influenced nothing, scored by the same "
          "code - agreement means the 100 were not a fluke.")


if __name__ == "__main__":
    main()
