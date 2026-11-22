"""The evaluation of Task 1, on the test split:   python evaluate_task1.py

Run it once, after train_task1.py. This is the only script in the folder that
reads the test split, and it reads it exactly once per run: no choice anywhere
in Task 1 was made on these 30 records, so nothing here needs a second chance.

It answers the four things the task asks to show:
  * the specification table, printed (spec.py)
  * the test Dice of every wave, P, QRS and T, next to the baseline's
  * one test record with the true and the predicted segmentation stacked
  * the QRS duration in ms of one test patient, predicted against annotated

Writes results/test_scores.json, figures/test_dice.png, figures/example_record.png
and prints all of it, so the numbers survive without the figures too.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

import baseline
import ludb_data
import metrics
import part1
import spec
from baseline import BaselineParams, predict_all
from ludb_data import CLASSES, FS, LENGTH, WAVES, Config
from part1 import UNet, predict

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"

# the numbers the task text quotes for the baseline, measured on its own basis
TASK_BASELINE = {"P": 0.74, "QRS": 0.89, "T": 0.85}
INK, MUTED, GRID = "#14161a", "#5b6068", "#e3e6ea"
C_MODEL, C_BASE = "#2a78d6", "#8a8f98"                     # colour does one job: the method
C_WAVE = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}        # and here: the wave


def load():
    chosen = json.loads((RESULTS / "choices.json").read_text(encoding="utf-8"))["chosen"]
    ckpt = torch.load(RESULTS / "model.pt", map_location=part1.device, weights_only=False)
    model = UNet(dim=1, c_in=1, c_out=4, widths=chosen["widths"]).to(part1.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    data = ludb_data.build(Config(**chosen["data"]))
    return chosen, model, data


def pick_record(data) -> int:
    """The record the figures show, fixed by a rule stated before any prediction:
    the test record with the most annotated QRS complexes, ties to the lowest
    record number. Choosing it by how well the model does on it would be
    cherry-picking, so we do not."""
    idx = data["split"]["test"]
    n_qrs = [(sum(1 for w in data["waves"][i] if w[0] == 2), int(data["record"][i]), int(i)) for i in idx]
    return max(n_qrs)[2]


def embed_whole_record(rows_pred, data, idx) -> np.ndarray:
    """Put predictions or labels made on the Option B arrays back where they came
    from in the whole 10 s record: sample *k* of row *r* becomes sample left+k,
    and every sample nobody annotated becomes class 0 (background)."""
    out = np.zeros((len(idx), LENGTH), np.int64)
    for r, i in enumerate(idx):
        left, valid = int(data["left"][i]), int(data["valid"][i])
        out[r, left:left + valid] = np.maximum(rows_pred[r][:valid], 0)   # -1 (unknown) -> 0 (background)
    return out


def naive_basis(data, idx, pred, base_pred) -> dict:
    """The task text's own label basis, so its quoted 0.74 / 0.89 / 0.85 can be
    read against ours. Nothing new is predicted and nothing is chosen: the same
    two label maps - passed in as the row blocks in idx order - the same records,
    only the treated-as-unknown samples turn into background. Outside the Option B
    crop both of our methods say background, while the task text's baseline
    searched the whole record and was penalised for waves it painted in the
    unannotated edges - so its own 0.74 / 0.89 / 0.85 are a harder version of our
    baseline row here, not the same one."""
    y = embed_whole_record(data["Y"][idx], data, idx)
    return {"baseline": metrics.evaluate(embed_whole_record(base_pred, data, idx), y),
            "model": metrics.evaluate(embed_whole_record(pred, data, idx), y)}


def cell(scores: dict, wave: str) -> str:
    """One cell of the per-rhythm table. "n/a" where nobody labelled that wave in
    that group - an atrial fibrillation record has no P wave to find, so a number
    there would be a number about nothing."""
    return f"{scores[wave]['per_record']:.2f}" if scores[wave]["labelled"] else "n/a"


def figure_dice(rows: dict, out: Path, n_records: int = 30) -> None:
    """Dice of every wave, our baseline against the model, with the task text's
    numbers for orientation. Bars start at 0; the value is printed on every bar;
    colour carries the method and nothing else."""
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    names = ["P", "QRS", "T"]
    width, xs = 0.34, range(len(names))
    base = [rows["baseline"][n]["global"] for n in names]
    model = [rows["model"][n]["global"] for n in names]
    task = [TASK_BASELINE[n] for n in names]
    ax.bar([x - width / 2 for x in xs], base, width, color=C_BASE, hatch="//", edgecolor="white",
           linewidth=1.2, label="baseline: R peaks + median distances, measured here")
    ax.bar([x + width / 2 for x in xs], model, width, color=C_MODEL, label="U-Net: 4 logits per sample, this model")
    ax.plot(list(xs), task, "o", ms=7, mfc="white", mec=INK, mew=1.4, linestyle="",
            label="the task text's baseline for that wave, on its own label basis *")
    for x, (b, m) in enumerate(zip(base, model)):
        ax.text(x - width / 2, b + 0.012, f"{b:.3f}", ha="center", fontsize=9, color=MUTED)
        ax.text(x + width / 2, m + 0.012, f"{m:.3f}", ha="center", fontsize=9, color=INK,
                fontweight="bold")
    ax.set_xticks(list(xs), [f"{n}\nin {rows['model'][n]['records_labelled']} of {n_records} records"
                             for n in names], fontsize=9)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("Dice, over the annotated samples", fontsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", labelsize=8, color=MUTED)
    # the legend goes under the axis labels, not over them: three rows, and the
    # two-line tick labels need their own band
    fig.tight_layout()
    fig.subplots_adjust(left=0.12, right=0.99, top=0.97, bottom=0.34)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 0.035), ncol=1)
    fig.text(0.02, 0.008, "* the task text's own basis: whole 10 s records, their unannotated edges "
                          "counted as background", fontsize=7, color=MUTED)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def trace(ax, y, valid, labels, title):
    """One record: the signal, and every sample coloured by its class. The grey
    band on the right is the padding that brings the cropped span back to 5000
    samples: no annotation there, so neither the loss nor the metric looks at it."""
    t = np.arange(LENGTH) / FS
    ax.axvspan(valid / FS, LENGTH / FS, color="0.90", zorder=0)
    for c in CLASSES:
        ax.fill_between(t, -3.2, 3.2, where=labels == c, color=C_WAVE[c], alpha=0.45, linewidth=0, zorder=1)
    ax.plot(t[:valid], y[:valid], color=INK, linewidth=0.7, zorder=3)   # the padding holds no signal, so draw none
    if LENGTH - valid > 250:                       # only where there is room for it
        ax.text(LENGTH / FS - 0.05, -3.0, "padding:\ntarget -1, not scored", ha="right",
                va="bottom", fontsize=7, color=MUTED, zorder=4)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xlim(0, LENGTH / FS)
    ax.set_yticks([])
    ax.set_title(title, fontsize=9, loc="left", color=INK)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def figure_record(data, i, pred_row, title_extra: str, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 4.6), sharex=True)
    valid = int(data["valid"][i])
    trace(axes[0], data["X"][i], valid, data["Y"][i],
          f"the cardiologists' labels   (record {data['record'][i]}, {data['rhythm'][i]})")
    trace(axes[1], data["X"][i], valid, pred_row, "what the network said" + title_extra)
    axes[1].set_xlabel("seconds (the record starts at its first annotated wave onset: Option B)", fontsize=8)
    axes[1].set_xticks(np.arange(0, 10.1, 1))
    axes[1].tick_params(labelsize=8)
    handles = [Patch(facecolor=C_WAVE[c], alpha=0.45, label=WAVES[c]) for c in CLASSES]
    fig.legend(handles=handles, fontsize=8, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    print(spec.text(), "\n")                          # the specification this run follows

    chosen, model, data = load()
    test = data["split"]["test"]
    print(f"test split: {len(test)} records, one patient each, chosen by the notebook's own "
          f"permutation(seed 0); this script is the only reader of it\n")

    # the baseline, on the same arrays and scored by the same function
    params = BaselineParams(**(chosen.get("baseline_params") or chosen["baseline"]["params"]))
    base_pred, offsets, _ = predict_all(data, params)
    print(baseline.describe_offsets(offsets))
    print(baseline.detection_report(data, params, "train") + "   (offsets measured here)")
    print(baseline.detection_report(data, params, "test") + "\n")

    out, target = predict(model, ludb_data.sets(data)["test"])
    pred = out.argmax(1).numpy()
    rows = {"baseline": metrics.evaluate(base_pred[test], data["Y"][test]),
            "model": metrics.evaluate(pred, data["Y"][test])}

    print("=" * 78)
    print("TEST SET - Dice of every wave, counted over the annotated samples only")
    print("=" * 78)
    reference = {n: {"global": TASK_BASELINE[n], "per_record": float("nan"), "records": 30}
                 for n in ("P", "QRS", "T")}
    print(metrics.table({"the task text's baseline *": reference,
                         "baseline (R peaks + median distances)": rows["baseline"],
                         "U-Net, this model": rows["model"]}))
    print("  * the numbers the task text quotes, from its own label basis: whole 10 s records in which")
    print("    the unannotated edges count as background. The two rows under it are measured on the")
    print("    Option B arrays - same records, same metric, only the annotated samples count. The")
    print("    comparison that means something is the U-Net against the baseline row above it.")
    print("\nglobal against per-record: " + "; ".join(
        f"{n} {rows['model'][n]['global']:.3f} / {rows['model'][n]['per_record']:.3f} "
        f"(labelled in {rows['model'][n]['records_labelled']} of {len(test)} records)"
        for n in ("P", "QRS", "T")))
    print("  a record with no labelled wave of some class adds nothing to the pooled global count;")
    print("  in per_record it counts as 0 there only if the method painted that wave anyway.")
    for name, s in rows.items():
        print(f"\nwhat the {name}'s Dice is made of on the test set:")
        print(metrics.balance_report(s))

    naive = naive_basis(data, test, pred, base_pred[test])
    print("\nthe same two label maps again on the task text's own basis - whole records, every")
    print("unannotated sample counted as background instead of unknown. Same predictions, same")
    print("records, same metric; only the denominator changed. Nothing was chosen on this table:")
    print(metrics.table({"baseline, task text's basis": naive["baseline"],
                         "U-Net, task text's basis": naive["model"]}))
    print(f"  the task text quotes {TASK_BASELINE['P']:.2f} / {TASK_BASELINE['QRS']:.2f} / "
          f"{TASK_BASELINE['T']:.2f} for its baseline, on this basis. Its detector looked for R")
    print("  peaks in the whole 10 s, so it also painted waves in the unannotated edges and was")
    print("  punished for them; ours only sees the cropped span, which makes our row the easier one.")

    # per rhythm: the test split is a mix, and a segmenter that only works on sinus
    # beats would still look respectable in a single number over all 30 records
    print("\nthe same Dice per rhythm group (mean over records; the n is small, so this is")
    print("a check that no group is failing, not a result about the rhythms themselves):")
    groups = {}
    for row, i in enumerate(test):
        groups.setdefault(data["rhythm"][i].split(",")[0], []).append(row)

    for rhythm, rws in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        s = metrics.evaluate(pred[rws], data["Y"][test][rws])
        b = metrics.evaluate(base_pred[test][rws], data["Y"][test][rws])
        print(f"  {rhythm:<24} n={len(rws):<3} model "
              + " ".join(f"{n} {cell(s, n)}" for n in ("P", "QRS", "T"))
              + "   baseline " + " ".join(f"{n} {cell(b, n)}" for n in ("P", "QRS", "T")))
    print("  (n/a: that rhythm's records carry no annotation for that wave at all - atrial")
    print("   fibrillation has no P wave - so there is no score to report for it)")

    # --- one test record, true above predicted, and its QRS duration ----------------
    i = pick_record(data)
    order = [int(j) for j in test]              # predict() hands the test set back in split order
    row = pred[order.index(i)]
    per_record = metrics.evaluate(row[None], data["Y"][i][None])
    true_ms, pred_ms = metrics.wave_durations_ms(row, data["Y"][i], 2, FS)
    pairs = metrics.matched_wave_durations_ms(row, data["Y"][i], 2, FS)
    extra = (f"  (its own Dice P {per_record['P']['global']:.2f} / QRS "
             f"{per_record['QRS']['global']:.2f} / T {per_record['T']['global']:.2f})")
    figure_record(data, i, row, extra, FIGURES / "example_record.png")

    print("\n" + "=" * 78)
    print(f"QRS DURATION OF ONE TEST PATIENT - record {data['record'][i]} "
          f"({data['rhythm'][i]}), {len(true_ms)} annotated QRS complexes")
    print("=" * 78)
    print("  annotated [ms]  " + "  ".join(f"{v:6.0f}" for v in true_ms)
          + f"   | median {np.median(true_ms):.0f}, mean {np.mean(true_ms):.1f}")
    print("  predicted [ms]  " + "  ".join(f"{v:6.0f}" for v in pred_ms)
          + f"   | median {np.median(pred_ms):.0f}, mean {np.mean(pred_ms):.1f}")
    if pairs:
        diff = np.array([p - t for t, p in pairs])
        print(f"  the {len(pairs)} annotated QRS the model also found: "
              f"median signed error {np.median(diff):+.0f} ms, mean |error| {np.mean(np.abs(diff)):.0f} ms, "
              f"worst {diff[np.argmax(np.abs(diff))]:+.0f} ms")
    print(f"  the network finds the QRS {len(pairs)}/{len(true_ms)} times within 60 ms of both edges")
    print("  (figures/example_record.png shows this record: labels above, prediction below)")

    figure_dice(rows, FIGURES / "test_dice.png", n_records=len(test))
    scores = {"test_records": [int(data["record"][i]) for i in test],
              "mean_dice_per_class": {n: rows["model"][n]["global"] for n in ("P", "QRS", "T")},
              "per_record_mean": {n: rows["model"][n]["per_record"] for n in ("P", "QRS", "T")},
              "baseline_mean_dice_per_class": {n: rows["baseline"][n]["global"] for n in ("P", "QRS", "T")},
              "task_text_baseline": TASK_BASELINE,
              "same_predictions_on_the_task_texts_basis": {
                  name: {n: round(naive[name][n]["global"], 3) for n in ("P", "QRS", "T")}
                  for name in naive},
              "counts": {n: rows["model"][n] for n in ("P", "QRS", "T")},
              "example_record": {"record": int(data["record"][i]), "rhythm": data["rhythm"][i],
                                 "annotated_qrs_ms": true_ms, "predicted_qrs_ms": pred_ms,
                                 "matched_pairs_ms": pairs,
                                 "dice": {n: per_record[n]["global"] for n in ("P", "QRS", "T")}},
              "config": chosen, "baseline_params": asdict(params)}
    (RESULTS / "test_scores.json").write_text(json.dumps(scores, indent=1), encoding="utf-8")
    print("\nwrote results/test_scores.json, figures/test_dice.png, figures/example_record.png")


if __name__ == "__main__":
    main()
