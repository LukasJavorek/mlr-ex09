"""The one look at the test split. Run it last, once.

    python evaluate_task3.py            # the real thing: reads split["test"]
    python evaluate_task3.py --valid    # the same code on the validation split, to
                                        # exercise it before that one read is spent

This is the **only file in the folder that reads `split["test"]`**, and it reads it
after the model is trained and every setting is fixed. `--valid` runs the identical
path on the 18 validation neurons and writes its output with `_valid` in the name:
plotting and printing code that has never been executed is exactly where a final
evaluation fails, and a failure halfway through would mean the test split had been
loaded for nothing. The dry run is what `run_all.py` does not do and RESULTS.md says
was done first.

What it reports, which is what the task asks to show:

* the network against both classical baselines, on the same 40 ms bins of the same
  neurons, each with a bootstrap interval over neurons - the yardstick being that one
  neuron is worth about +-0.03 and the mean of 34 about +-0.005;
* the mean over neurons (what the challenge scores) beside the mean over the ten
  datasets (a different number: the datasets hold 1 to 12 test neurons apiece);
* the same per dataset, since a model that wins on average can still meet a dataset
  where the derivative baseline loses to the fluorescence (dataset 4);
* how many predictions came out constant, i.e. undefined, and the two controls - the
  same predictions rolled through time, and scored against another neuron's spike
  train;
* the gap over the baseline with the sd of its **paired** bootstrap, which is far
  tighter than the two intervals subtracted because both methods share their neurons;
* and, on the test split only, the score with and without neuron (9, 4) - the test
  copy of the neuron dataset 9 contains twice, whose training copy this pipeline
  drops. Both numbers come out of the one read and neither is chosen between.

Everything is `metrics.score` over whole recordings: every finite bin of every neuron
in the split, no windows and no padding, the same call that scored the validation
split during training.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import baseline
import figs
import fitting
import metrics
import sf_data
import spec
from validate_choices import CHOICES, RESULTS

NETWORK = "network (this model)"


def main(split: str = "test") -> None:
    sys.stdout.reconfigure(line_buffering=True)      # so `| tee results/eval_log.txt` is live
    dry = split != "test"
    tag = "_valid" if dry else ""
    print(spec.text())
    if not (RESULTS / "model.pt").exists():
        raise SystemExit("no results/model.pt - run `python train_task3.py` first")
    chosen = json.loads(CHOICES.read_text(encoding="utf-8"))["chosen"]
    cfg = sf_data.Config(**chosen["data"])
    bins, k = cfg.metric_bins, cfg.decimate

    sf_data.seed_everything(chosen["seed"])
    data = sf_data.build(cfg, verbose=False)
    neurons, truth = data["split"][split], data["y"]
    hours = sum(len(truth[n]) for n in neurons) * k / sf_data.FS_IN / 3600
    banner = ("DRY RUN on the validation split. The test split is not loaded by this run."
              if dry else
              "READING THE TEST SPLIT - the first and only time this pipeline looks at it.")
    print(f"\n{'=' * 100}\n{banner}\n"
          f"{len(neurons)} neurons, {hours:.1f} h of recording, "
          f"{sum(len(truth[n]) for n in neurons) * bins / 1000:.0f} thousand 40 ms bins.\n"
          + "=" * 100)

    model = fitting.load_model(chosen["widths"], RESULTS / "model.pt")
    preds = fitting.predict_split(model, data, split)
    raw = baseline.traces()
    sigma = chosen["baseline_sigma"]
    deriv = f"derivative, sigma {sigma}"
    methods = {NETWORK: preds,
               deriv: {n: baseline.derivative(raw[n], sigma, k) for n in neurons},
               "fluorescence": {n: baseline.fluorescence(raw[n], k) for n in neurons}}
    per = {name: {n: metrics.correlate(p[n], truth[n], bins) for n in neurons}
           for name, p in methods.items()}
    scores = {name: metrics.score(v) for name, v in per.items()}

    print(f"\nthe challenge's metric: 40 ms bins, Pearson per neuron, mean over neurons "
          f"with a bootstrap 95% interval")
    for name in methods:
        print(metrics.summarize(name, scores[name]))
    controls = metrics.control_means(preds, {n: truth[n] for n in neurons}, bins=bins)
    print(f"{'controls on the network':<26} rolled through time {controls['shifted']:+.3f}   "
          f"another neuron's spikes {controls['shuffled']:+.3f}")

    gap = scores[NETWORK]["mean_over_neurons"] - scores[deriv]["mean_over_neurons"]
    sd = paired_gap_sd(per[NETWORK], per[deriv])
    floor = 0.03 / np.sqrt(len(neurons))      # one neuron is worth ~+-0.03; the audit measured it
    print(f"\nnetwork minus derivative baseline: {gap:+.3f} (paired bootstrap sd {sd:.3f}, "
          f"{gap / sd:+.1f} sd). Both are scored on the same {len(neurons)} neurons, so the "
          f"paired sd is much tighter than the two intervals above; and remember a mean over "
          f"{len(neurons)} neurons moves ~{floor:.3f} on luck alone, so a gap under that is "
          f"not a result.")

    print(f"\nper dataset - * marks a dataset where the derivative is not even the better "
          f"of the two baselines:")
    print(f"{'dataset':>8} {'neurons':>7} {'network':>8} {'deriv':>7} {'fluor':>7} "
          f"{'worst neuron':>12}")
    per_ds = {}
    for ds in sf_data.DATASETS:
        rows = [n for n in neurons if n[0] == ds]
        if not rows:
            continue
        one = {name: scores[name]["per_dataset"].get(str(ds), float("nan")) for name in methods}
        worst = min(v for n, v in per[NETWORK].items() if n[0] == ds and v is not None)
        per_ds[ds] = {"n": len(rows), **{name: round(one[name], 4) for name in methods},
                      "worst_neuron": round(float(worst), 4)}
        star = " *" if one[deriv] < one["fluorescence"] else ""
        print(f"{ds:>8} {len(rows):>7} {one[NETWORK]:>8.3f} {one[deriv]:>7.3f} "
              f"{one['fluorescence']:>7.3f} {worst:>12.3f}{star}")

    won = [ds for ds in per_ds if per_ds[ds][NETWORK] > per_ds[ds][deriv]]
    lost = [ds for ds in sf_data.DATASETS if ds not in per_ds or per_ds[ds][NETWORK]
            <= per_ds[ds][deriv]]
    per_ds_won = {"n_datasets": len(per_ds), "won": won, "not_won": lost}
    print(f"\nthe task asks to beat the derivative on average **and in most of the ten "
          f"datasets**: it does in {len(won)} of {len(per_ds)}"
          + (f" (not in {lost}, where the network is worse by "
             f"{max(per_ds[ds][deriv] - per_ds[ds][NETWORK] for ds in lost):.3f} - read that "
             f"against the +-0.03 worth one neuron, and the per-dataset means in these "
             f"datasets rest on as few as one neuron)" if lost else "."))

    twin, rest = None, None
    if not dry:
        twin, rest = per[NETWORK].get((9, 4)), [v for n, v in per[NETWORK].items()
                                                if n != (9, 4) and v is not None]
        print(f"\nthe duplicated neuron: test neuron (9,4) scores "
              f"{'undefined' if twin is None else f'{twin:.3f}'}; its training twin (9,3) was "
              f"dropped, so nothing here has seen it. Mean over the other {len(rest)} test "
              f"neurons {np.mean(rest):.3f}, over all {len(neurons)} "
              f"{scores[NETWORK]['mean_over_neurons']:.3f}.")

    RESULTS.mkdir(exist_ok=True)
    out = {"split": split, "config": asdict(cfg), "chosen": chosen, "bins_ms": 40,
           "n_neurons": len(neurons), "hours_of_signal": round(hours, 2),
           "scores": scores, "controls": controls, "per_dataset": per_ds,
           "beats_derivative": per_ds_won,
           "per_neuron": {f"{n[0]},{n[1]}": (None if v is None else round(v, 4))
                          for n, v in per[NETWORK].items()},
           "gap_over_derivative": round(float(gap), 4), "gap_paired_bootstrap_sd": round(sd, 4)}
    if rest is not None:
        out["mean_excluding_9_4"] = round(float(np.mean(rest)), 4)
    path = RESULTS / f"test_scores{tag}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path.relative_to(RESULTS.parent)}"
          + ("   <- the dry run's file; the real one is results/test_scores.json" if dry else ""))

    figure(scores, per, deriv, len(neurons), split, tag)
    example(data, preds, truth, raw, sigma, tag)


def paired_gap_sd(a: dict, b: dict, draws: int = 2000) -> float:
    """Bootstrap sd of the per-neuron difference network - baseline. Paired, because
    both methods are scored on the same neurons: the noise that a hard neuron brings
    one method it brings the other, and it cancels."""
    d = np.array([x - y for x, y in zip(a.values(), b.values())
                  if x is not None and y is not None], dtype=np.float64)
    rng = np.random.default_rng(0)
    return float(np.std(rng.choice(d, size=(draws, len(d)), replace=True).mean(1)))


def figure(scores, per, deriv, n: int, split: str, tag: str) -> None:
    """Left: the three methods with their intervals. Right: every neuron in the split,
    network and baseline side by side, so the spread the mean averages away is visible -
    this metric is a mean over a short, wide list, and the list is the finding."""
    figs.style()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    names = list(scores)
    for i, name in enumerate(names):
        s, y = scores[name], len(names) - 1 - i
        lo, hi = s["boot95_over_neurons"]
        ax.errorbar(s["mean_over_neurons"], y,
                    xerr=[[s["mean_over_neurons"] - lo], [hi - s["mean_over_neurons"]]],
                    fmt="o", color=figs.SERIES[i], ms=6, capsize=3, lw=1.2, label=name)
        ax.annotate(f"{s['mean_over_neurons']:.3f}", (hi + 0.015, y), fontsize=7, va="center",
                    color=figs.INK)
    ax.set_yticks(range(len(names))[::-1])
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("correlation with the true spike counts")
    ax.set_title(f"mean over the {n} {split} neurons", loc="left", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower right", frameon=True, facecolor=figs.SURFACE,
              edgecolor=figs.GREY, framealpha=0.92)

    ax = axes[1]
    order = sorted((k for k, v in per[NETWORK].items() if v is not None), key=lambda k: per[NETWORK][k])
    for i, key in enumerate(order):
        ax.plot([per[deriv][key], per[NETWORK][key]], [i, i], color=figs.GREY, lw=0.7, zorder=2)
        ax.plot(per[deriv][key], i, "o", ms=4, color=figs.GREEN, zorder=3)
        ax.plot(per[NETWORK][key], i, "o", ms=4.5, color=figs.BLUE, zorder=3)
    ax.axvline(0, color=figs.MUTED, lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{k[0]},{k[1]}" for k in order], fontsize=5)
    ax.set_xlabel("correlation, per neuron (rows sorted by the network)")
    ax.set_ylabel("neuron  (dataset,column)")
    ax.set_title("every neuron: the mean is one number out of this list", loc="left",
                 fontsize=9)
    handles = [plt.Line2D([], [], color=figs.BLUE, marker="o", lw=0, ms=4.5, label="network"),
               plt.Line2D([], [], color=figs.GREEN, marker="o", lw=0, ms=4, label="derivative")]
    ax.legend(handles=handles, fontsize=6.5, loc="upper left", frameon=True,
              facecolor=figs.SURFACE, edgecolor=figs.GREY, framealpha=0.92)
    fig.tight_layout()
    figs.save(fig, f"test_scores{tag}.png")


def example(data, preds, truth, raw, sigma, tag: str, seconds: int = 30) -> None:
    """Half a minute of one neuron - the one whose score sits nearest the median of the
    split, so it is typical rather than flattering, and always the **first** 30 s of its
    recording rather than a window picked after looking at it. What it is given, what it
    answers, and what it is scored against. Both continuous series are rescaled to their
    own range and labelled as such: the metric is affine invariant, so the shape is the
    whole content and the absolute rate is not even visible to it."""
    figs.style()
    cfg = data["cfg"]
    scored = {k: metrics.correlate(preds[k], truth[k], cfg.metric_bins) for k in preds}
    valid = [k for k, v in scored.items() if v is not None]
    key = min(valid, key=lambda k: abs(scored[k] - np.median([scored[k] for k in valid])))
    n = min(int(seconds * cfg.fs), len(truth[key]))
    t = np.arange(n) / cfg.fs
    x, y, p = data["x"][key][:n], truth[key][:n], preds[key][:n]
    d = baseline.derivative(raw[key], sigma, cfg.decimate)[:n]
    rescale = lambda v: (v - v.min()) / (v.max() - v.min() + 1e-9)             # noqa: E731
    top = max(float(np.nanmax(y)), 1.0)

    fig, axes = plt.subplots(2, 1, figsize=(7.6, 4.2), sharex=True,
                             gridspec_kw={"height_ratios": [1, 0.85]})
    ax = axes[0]
    ax.plot(t, rescale(x), color=figs.GREY, lw=1.0, label="fluorescence: the only input")
    ax.plot(t, rescale(p), color=figs.BLUE, lw=1.3, label="predicted spike rate, rescaled")
    ax.set_ylabel("arbitrary units")
    ax.set_title(f"neuron {key[0]},{key[1]} - scores {scored[key]:.3f}, the median of the "
                 f"split; the first {n / cfg.fs:.0f} s of a "
                 f"{len(truth[key]) * cfg.decimate / sf_data.FS_IN:.0f} s recording",
                 loc="left", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower left", frameon=True, facecolor=figs.SURFACE,
              edgecolor=figs.GREY, framealpha=0.92)
    ax = axes[1]
    ax.step(t, y, where="post", color=figs.ORANGE, lw=1.0, label="true spikes per 40 ms bin")
    ax.plot(t, rescale(d) * top, color=figs.GREEN, lw=0.9, ls=(0, (2, 1)),
            label="derivative baseline, rescaled to the same axis")
    ax.set_xlabel("seconds from the start of the recording; one step = 40 ms")
    ax.set_ylabel("spikes per bin")
    ax.set_yticks(range(0, int(top) + 1))
    ax.legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    figs.save(fig, f"example_neuron{tag}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--valid", action="store_true",
                    help="dry run on the validation split; does not read the test split")
    main("valid" if ap.parse_args().valid else "test")
