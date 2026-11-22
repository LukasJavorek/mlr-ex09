"""Every choice Task 1 makes, made on the validation split - run it before
training the final model:

    python validate_choices.py            # stages A-D,   ~16 min on a 6-thread CPU
    python validate_choices.py --refine   # stages E and F, ~36 min
    python validate_choices.py --confirm  # stage G,      ~54 min
    python validate_choices.py --quick    # the same stages, fewer settings: ~9 + 11
                                          # + 12 min, and stage G's 150-epoch run
                                          # (the most expensive in Task 1) is skipped

The test split is never read here. Stages, each judged by the mean of the three
validation Dice (P, QRS, T), pooled over the annotated samples:

  A  the baseline's detector thresholds            - 120 settings, no training
  B  how the record is cropped and scaled          - margin x standardization
  C  what the loss weighs                          - none x inverse frequency x squared
  D  the network and the learning rate             - widths x lr
  E  how long it trains, and how deep the U-Net is - separately, because the first
       sweep stops while its validation loss is still falling:
                                                    python validate_choices.py --refine
  G  the deeper networks re-run at the length the final run trains for, because E
       and F were measured on different rulers:
                                                    python validate_choices.py --confirm

Writes results/choices.json and figures/validation_choices.png. train_task1.py
takes the winners from results/choices.json, so the final run is reproducible
from this file alone.
"""
from __future__ import annotations

import itertools
import json
import textwrap
import time
from dataclasses import asdict
from pathlib import Path

import warnings

warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

import matplotlib
matplotlib.use("Agg")                            # no window; the figures go to figures/*.png
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import ludb_data
import metrics
import part1
from baseline import BaselineParams, measure_offsets, predict_record, r_peaks
from ludb_data import CLASSES, Config, WAVES, seed_everything
from part1 import UNet, predict, train

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
EPOCHS = 25                                      # train() keeps the best validation epoch anyway
MUTED, GRID = "#5b6068", "#e3e6ea"               # the same two greys the other figures use


def val_scores(model, data) -> dict:
    model.eval()
    out, target = predict(model, ludb_data.sets(data)["valid"])
    return metrics.evaluate(out.argmax(1).numpy(), target.numpy())


def mean_dice(scores) -> float:
    return float(np.mean([scores[WAVES[c]]["global"] for c in CLASSES]))


def _run(data, weights: str, widths: list, lr: float, epochs: int, batch_size: int = 16) -> dict:
    """One training run of the whole pipeline, scored on the validation split."""
    seed_everything(0)
    model = UNet(dim=1, c_in=1, c_out=4, widths=widths)
    w = None if weights == "none" else torch.tensor(
        ludb_data.class_weights(data, power=float(weights)), dtype=torch.float32)
    sets = ludb_data.sets(data)
    t0 = time.perf_counter()
    model = train(model, sets["train"], sets["valid"],
                  nn.CrossEntropyLoss(ignore_index=-1, weight=w), epochs=epochs, lr=lr,
                  batch_size=batch_size)
    scores = val_scores(model, data)
    seconds = time.perf_counter() - t0
    plt.close("all")
    return {"mean": mean_dice(scores),
            "dice": {name: round(scores[name]["global"], 3) for name in ("P", "QRS", "T")},
            "balance": {name: [round(scores[name]["recall"], 2), round(scores[name]["precision"], 2)]
                        for name in ("P", "QRS", "T")},
            "seconds": seconds}


def stage(name: str, variants: list) -> list:
    """Runs every variant of one stage, prints and sorts them by mean validation Dice.
    variants = [(label, what to record in the json, kwargs for _run), ...]."""
    print(f"\n{name}  (judged on the validation split)", flush=True)
    rows = []
    for label, record, kwargs in variants:
        r = _run(**kwargs)
        rows.append({"label": label, **record, **r, "seconds": round(r["seconds"], 1)})
        print(f"    {label:<30} mean {r['mean']:.3f}   "
              + "  ".join(f"{w} {r['dice'][w]:.3f}" for w in ("P", "QRS", "T"))
              + f"   [{r['seconds']:.0f} s]", flush=True)
    rows.sort(key=lambda row: -row["mean"])
    return rows


def refine(quick: bool = False) -> None:
    """Stages E and F, run after the main sweep:   python validate_choices.py --refine

    Two questions the first sweep leaves open, both answered on the validation
    split like every other choice:

    E  how long to train. Stages B-D stop at 25 epochs with the validation loss
       still falling, so "longer" is an untested guess rather than a result.
       train() keeps the best validation epoch, so a longer run can only match or
       beat a shorter one on that epoch; what it costs is wall-clock time, and
       what buys the time back is a smaller batch (more optimizer steps per epoch,
       which is what a dataset of 140 records needs).

    F  how deep. Stage D found depth to be the biggest lever of all - one more
       pooling level, hence one more scale of context, moved the mean Dice further
       than the class weights, the margin and the learning rate together - so the
       neighbours of the winner are worth their runs. A deeper net is only taken if
       it beats the incumbent by more than the noise 30 records can produce.
    """
    results = json.loads((RESULTS / "choices.json").read_text(encoding="utf-8"))
    chosen = results["chosen"]
    chosen.setdefault("baseline_params", chosen["baseline"]["params"])   # files written before that key
    data = ludb_data.build(Config(**chosen["data"]))
    panels = results.setdefault("stages", {})
    common = dict(data=data, weights=chosen["weights"], lr=chosen["lr"])

    variants = [(f"{ep} epochs, batch {bs}", {"epochs": ep, "batch_size": bs},
                 dict(**common, widths=chosen["widths"], epochs=ep, batch_size=bs))
                for ep, bs in ([(75, 16)] if quick else [(75, 16), (150, 16), (75, 8)])]
    rows = stage("stage E - how long the training runs", variants)
    panels["E epochs and batch"] = [{k: v for k, v in r.items() if k != "seconds"} for r in rows]
    chosen["epochs"], chosen["batch_size"] = rows[0]["epochs"], rows[0]["batch_size"]
    incumbent_full = rows[0]["mean"]          # the current network, at the chosen length

    nets = [[16, 32, 64, 128, 256]] if quick else [[16, 32, 64, 128, 256], [32, 64, 128, 256, 512]]
    variants = [(f"widths {w}", {"widths": w}, dict(**common, widths=w, epochs=EPOCHS, batch_size=16))
                for w in nets]
    rows_f = stage(f"stage F - deeper still, at the sweep's {EPOCHS} epochs", variants)
    panels["F depth (at the sweep's 25 epochs)"] = [
        {k: v for k, v in r.items() if k != "seconds"} for r in rows_f]
    incumbent = _run(data, chosen["weights"], chosen["widths"], chosen["lr"], EPOCHS, 16)
    print(f"    the incumbent {chosen['widths']} re-measured: mean {incumbent['mean']:.3f}")
    # a deeper network converges more slowly, so winning at the sweep's 25 epochs is
    # only half an answer: it has to win at the length the run actually trains for too
    if rows_f and rows_f[0]["mean"] > incumbent["mean"] + 0.01:
        print(f"    stage F likes {rows_f[0]['widths']} at {EPOCHS} epochs - confirming it at "
              f"{chosen['epochs']}, against the incumbent's {incumbent_full:.3f}", flush=True)
        confirm = _run(data, chosen["weights"], rows_f[0]["widths"], chosen["lr"],
                       chosen["epochs"], chosen["batch_size"])
        panels["F confirmed at full length"] = [
            {"label": f"widths {rows_f[0]['widths']}", "mean": confirm["mean"]},
            {"label": f"widths {chosen['widths']} (incumbent)", "mean": incumbent_full}]
        print(f"    {rows_f[0]['widths']} at {chosen['epochs']} epochs: mean {confirm['mean']:.3f}")
        if confirm["mean"] > incumbent_full:
            chosen["widths"] = rows_f[0]["widths"]

    (RESULTS / "choices.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    figure(panels)
    print(f"\nchosen: widths {chosen['widths']}, {chosen['epochs']} epochs, batch "
          f"{chosen['batch_size']} -> results/choices.json, figures/validation_choices.png")


def confirm_depth(quick: bool = False) -> None:
    """Stage G, run after --refine:   python validate_choices.py --confirm

        python validate_choices.py --confirm

    Stage F judged every depth at the sweep's 25 epochs and found five pooling
    levels to be worth about +0.035 over the four it had chosen there. A number
    measured at 25 epochs is not the same number at the length the final run trains
    for, and refine() switching on it alone would have been a choice made on the
    wrong ruler - so both candidates run again at the chosen length, next to the
    incumbent's own score at that length, which stage E already measured.

    Two more things are decided here rather than assumed:

    * two candidates within 0.005 of each other: the smaller network wins. Equal
      Dice on 30 validation records is not a reason to train three times as long,
      and the smaller one is the one a later reader can reproduce on a laptop.
    * how long the winner trains: stage E answered that for the four-level network,
      not for this one, so the winner runs at twice the length too and is only kept
      if it is better there. Under --quick that last run is skipped - it is the
      single most expensive training in Task 1 (a 5-level network at 150 epochs) and
      --quick is the preview mode, so it keeps the winner at the chosen length
      instead of proving that length is not too short.
    """
    results = json.loads((RESULTS / "choices.json").read_text(encoding="utf-8"))
    chosen, panels = results["chosen"], results.setdefault("stages", {})
    data = ludb_data.build(Config(**chosen["data"]))

    row_e = next(r for r in panels["E epochs and batch"]
                 if r["epochs"] == chosen["epochs"] and r["batch_size"] == chosen["batch_size"])
    incumbent = {"label": f"widths {chosen['widths']}, {chosen['epochs']} epochs",
                 "widths": chosen["widths"], "mean": row_e["mean"],
                 "dice": row_e["dice"], "epochs": chosen["epochs"]}
    print(f"stage G - the deeper networks at the length the run trains for "
          f"({chosen['epochs']} epochs), judged on the validation split")
    print(f"    the incumbent {chosen['widths']} at {chosen['epochs']} epochs: mean {incumbent['mean']:.3f}")

    depth = panels.get("F depth (at the sweep's 25 epochs)", panels.get("F depth", []))
    cands = [r["widths"] for r in depth if r["widths"] != chosen["widths"]]
    cands = cands[:1] if quick else cands[:2]
    variants = [(f"widths {w}, {chosen['epochs']} epochs", {"widths": w, "epochs": chosen["epochs"]},
                 dict(data=data, weights=chosen["weights"], lr=chosen["lr"], widths=w,
                      epochs=chosen["epochs"], batch_size=chosen["batch_size"])) for w in cands]
    rows = stage("stage G", variants)
    rows.append(incumbent)
    rows.sort(key=lambda r: -r["mean"])
    within = [r for r in rows if r["mean"] >= rows[0]["mean"] - 0.005]
    best = min(within, key=lambda r: sum(r["widths"]))         # a near-tie goes to the smaller net
    print(f"    within 0.005 of the top: {[r['widths'] for r in within]} -> the smaller one, {best['widths']}")

    if quick:
        keep = best
        print(f"    --quick: skipping the {2 * best['epochs']}-epoch confirmation run, "
              f"keeping {best['widths']} at {best['epochs']} epochs")
    else:
        longer = 2 * best["epochs"]
        r_long = _run(data, chosen["weights"], best["widths"], chosen["lr"], longer, chosen["batch_size"])
        rows.append({"label": f"widths {best['widths']}, {longer} epochs", "widths": best["widths"],
                     "mean": r_long["mean"], "dice": r_long["dice"], "epochs": longer})
        rows.sort(key=lambda r: -r["mean"])
        keep = best if best["mean"] >= r_long["mean"] else {**best, "epochs": longer}
    panels["G at full length"] = [{k: v for k, v in r.items() if k != "seconds"} for r in rows]
    chosen["widths"], chosen["epochs"] = keep["widths"], keep["epochs"]

    (RESULTS / "choices.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    figure(panels)
    print("\nthe whole stage G table (mean Dice of P, QRS, T on the validation split):")
    for r in rows:
        print(f"    {r['label']:<44} mean {r['mean']:.3f}   "
              + "  ".join(f"{w} {r['dice'][w]:.3f}" for w in ("P", "QRS", "T")))
    print(f"\nchosen: widths {chosen['widths']}, {chosen['epochs']} epochs, batch "
          f"{chosen['batch_size']} -> results/choices.json, figures/validation_choices.png")


# --------------------------------------------------------------------------- #
def sweep_baseline(data) -> tuple[dict, list]:
    """Stage A: the baseline's three thresholds, on the validation split only."""
    offsets = measure_offsets(data)
    idx = data["split"]["valid"]
    grid = []
    for band, dist, keep, anchor in itertools.product(
            [(5., 15.), (8., 20.)], [200, 240, 300], [0.2, 0.35, 0.5, 0.65, 0.8],
            ["max", "p90", "p75", "median"]):
        p = BaselineParams(band_hz=band, distance_ms=dist, keep_rel=keep, anchor=anchor)
        pred = np.zeros_like(data["Y"])
        for i in idx:
            valid = int(data["valid"][i])
            pred[i, :valid] = predict_record(data["X"][i], valid, offsets, p)
        scores = metrics.evaluate(pred[idx], data["Y"][idx])
        grid.append({"params": asdict(p), "dice": {w: round(scores[w]["global"], 3) for w in ("P", "QRS", "T")},
                     "mean": mean_dice(scores)})
    grid.sort(key=lambda g: -g["mean"])
    return grid[0], grid


def figure(stages: dict) -> None:
    """One panel per stage: the mean validation Dice of every variant. Lollipops
    rather than bars - the variants differ by 0.01 to 0.1 and a bar chart drawn from
    0 would show eight identical lengths. Two columns, because one panel per stage
    in a single row is unreadable once the labels are legible."""
    items = sorted(stages.items())          # by stage letter, whatever order they were added in
    ncols, nrows = 2, -(-len(items) // 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 2.7 * nrows), squeeze=False)
    for ax, (title, rows) in zip(axes.ravel(), items):
        rows = sorted(rows, key=lambda r: r["mean"])
        labels = ["\n".join(textwrap.wrap(str(r["label"]), 26)) for r in rows]
        y = np.arange(len(rows))
        mean = [r["mean"] for r in rows]
        ax.hlines(y, 0.5, mean, color="#2a78d6", linewidth=1.4)
        ax.plot(mean, y, "o", color="#2a78d6", ms=5)
        for yy, m in zip(y, mean):
            ax.text(m + 0.007, yy, f"{m:.3f}", va="center", fontsize=7)
        ax.set_yticks(y, labels, fontsize=7)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlim(0.5, 1.0)
        ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
        ax.tick_params(axis="x", labelsize=7, color=MUTED)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    fig.suptitle("every choice of Task 1, judged by the mean Dice of P, QRS and T on the "
                 "validation split (the test split is never read here)", fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGURES / "validation_choices.png", dpi=150)
    plt.close(fig)


def main(quick: bool = False) -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    chosen: dict = {}
    panels: dict = {}
    trained = dict(weights="none", widths=[32, 64, 128], lr=1e-3, epochs=EPOCHS, batch_size=16)

    # A -------------------------------------------------------------------- #
    print("stage A - the baseline's detector thresholds (no training, "
          "judged on the validation split)")
    data = ludb_data.build(Config())
    best, grid = sweep_baseline(data)
    print(f"    best of {len(grid)}: {best['params']} -> {best['dice']}, mean {best['mean']:.3f}")
    chosen["baseline"] = best
    chosen["baseline_params"] = best["params"]     # what baseline.BaselineParams takes, flat
    panels["A baseline band"] = [
        {"label": f"{band[0]:.0f}-{band[1]:.0f} Hz",
         # asdict() leaves a tuple a tuple: compare like with like, or this filter
         # matches nothing and the panel fills with nan
         "mean": float(np.mean([g["mean"] for g in grid if tuple(g["params"]["band_hz"]) == band])),
         "n": sum(tuple(g["params"]["band_hz"]) == band for g in grid)}
        for band in ((5.0, 15.0), (8.0, 20.0))]
    for row in panels["A baseline band"]:
        assert np.isfinite(row["mean"]) and row["n"], row    # a nan means the filter matched nothing
        print(f"    band {row['label']:<10} mean over its {row['n']} settings {row['mean']:.3f}", flush=True)

    # B -------------------------------------------------------------------- #
    cfgs = [Config(0, "span"), Config(0, "record")]
    if not quick:
        cfgs += [Config(128, "span"), Config(250, "span")]
    built, variants = {}, []
    for cfg in cfgs:
        built[cfg.key()] = ludb_data.build(cfg)          # a margin or scaling change rebuilds the arrays
        variants.append((f"margin {cfg.margin}, {cfg.standardize} statistics", {"config": asdict(cfg)},
                         dict(data=built[cfg.key()], **{k: v for k, v in trained.items() if k != "weights"},
                              weights="none")))
    rows_b = stage("stage B - how the record is cropped and scaled", variants)
    panels["B crop and scaling"] = rows_b
    chosen["data"] = rows_b[0]["config"]

    # C -------------------------------------------------------------------- #
    data = built[Config(**chosen["data"]).key()]
    powers = [] if quick else ["1.0"]
    variants = [("no class weights", {"weights": "none"},
                 dict(data=data, **{**trained, "weights": "none"}))]
    variants += [(f"inverse frequency to the {p}", {"weights": p},
                  dict(data=data, **{**trained, "weights": p})) for p in ["0.5"] + powers]
    rows_c = stage("stage C - what the loss weighs", variants)
    panels["C loss weighting"] = rows_c
    chosen["weights"] = rows_c[0]["weights"]

    # D -------------------------------------------------------------------- #
    nets = [([32, 64, 128], 1e-3), ([32, 64, 128], 3e-3)]
    if not quick:
        nets += [([16, 32, 64, 128], 1e-3), ([32, 64, 128, 256], 1e-3)]
    variants = [(f"widths {w}, lr {lr:g}", {"widths": w, "lr": lr},
                 dict(data=data, **{**trained, "weights": chosen["weights"], "widths": w, "lr": lr}))
                for w, lr in nets]
    rows_d = stage("stage D - the network and the learning rate", variants)
    panels["D network and lr"] = rows_d
    chosen["widths"], chosen["lr"] = rows_d[0]["widths"], rows_d[0]["lr"]

    chosen["batch_size"] = 16
    chosen["epochs"] = 3 * EPOCHS            # train() keeps the best validation epoch, so more is safe
    chosen["seed"] = 0
    (RESULTS / "choices.json").write_text(
        json.dumps({"chosen": chosen, "baseline_grid": grid, "stages": panels}, indent=1, default=str),
        encoding="utf-8")
    figure(panels)
    print(f"\nchosen for the final run: {json.dumps(chosen)}")
    print("wrote results/choices.json and figures/validation_choices.png")


if __name__ == "__main__":
    import sys

    if "--refine" in sys.argv:
        refine(quick="--quick" in sys.argv)
    elif "--confirm" in sys.argv:
        confirm_depth(quick="--quick" in sys.argv)
    else:
        main(quick="--quick" in sys.argv)
