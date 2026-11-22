"""Every choice Task 3 makes, made on the **validation** split and nowhere else.

    python validate_choices.py            # stage 1: the data
    python validate_choices.py --refine   # stage 2: lr, depth, batch, then how long
    python validate_choices.py --confirm  # stage 3: the winner at full length, twice

writes `results/choices.json`, which is what `sf_data.configured()`, `train_task3.py`
and `evaluate_task3.py` read. Nothing here opens the test split - `main` asserts that
no test neuron even has a target scale, which is the shape a leak would take.

**Arms are scored by the challenge's own correlation over whole validation recordings,
averaged over 3 seeds.** One validation neuron is worth about +-0.03 and the 18-neuron
mean about +-0.045, which is larger than most of the differences between neighbouring
settings; a single run per arm would rank noise. Each arm is also scored at *its own
best epoch within the budget* (`fitting.fit` keeps those weights), so an arm that
converges slowly is not punished for the budget - and the epoch at which its best fell
is printed beside it, so an arm that was still climbing when the budget ran out is
visible as such.

Stage 1 asks the questions the data audit raised, one field of the Config at a time:

  decimate    1, 2, 4 released samples per input sample (100, 50, 25 Hz), window held at
              the same 20.5 s: the same windows and the same number of gradient steps,
              so the score difference is what the coarse grid buys, and the coarse grid
              also multiplies the receptive field per layer by 4
  window      128, 256, 1024 samples at 25 Hz (5, 10, 41 s) on top of the default 512:
              context per window against number of windows
  standardize none, i.e. the released fluorescence with its 13x spread in std
  scale       the target divided by each neuron's own mean rate (the spec) or by one
              global rate - the direct test of the audit's weighting argument

Every sweep chooses by one rule, in `pick()`: the best candidate (ablations of the
specification are measured, not selected), and among the arms within 0.005 of the best -
a gap three seeds cannot resolve - the cheapest for lr and batch, and for depth the arm
whose receptive field fits the calcium kernel and the window. `--rechoose` re-applies the
rule to arms already trained, without training anything.

Stage 2 takes that winner and tunes lr, the number of levels, and the batch size, then
reads the epoch count off a long curve. Stage 3 re-runs the winner and the runner-up at
full length; puts the dataset 9 duplicate back in the training set, which on validation
changes nothing because its twin is a test neuron - exactly why the leak had to be closed
by reasoning rather than by measurement; and separates the two things a short window buys
at once, context per window and updates per epoch, by re-running the short-window winner
with its batch raised by the window ratio - the same updates per epoch - beside a
512-sample window at the same length.

Costs measured on a 6-thread CPU (one epoch: 1.5 s at 3 levels, 2.2 s at 4, 3.6 s at 5
on the 25 Hz grid; the 100 Hz arms are 4x that): stage 1 ~15 min, stage 2 ~25 min,
stage 3 ~20 min. `--epochs` overrides the per-arm budget; `--seeds 1` makes everything
a third of it and the rankings correspondingly noisier.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

import fitting
import metrics
import part1
import sf_data
from sf_data import Config, seed_everything

HERE = sf_data.HERE
RESULTS = HERE / "results"
CHOICES = RESULTS / "choices.json"

SHORT = 16                       # epochs per arm in stages 1 and 2
LONG = 90                        # the epoch curve in --refine, and the runs in --confirm
WIDTHS = [16, 32, 64]            # the architecture every stage-1 arm shares


# --------------------------------------------------------------------------- #
def load() -> dict:
    return json.loads(CHOICES.read_text(encoding="utf-8")) if CHOICES.exists() else {"sweeps": {}}


def save(state: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    CHOICES.write_text(json.dumps(state, indent=2, sort_keys=False), encoding="utf-8")
    print(f"wrote {CHOICES.relative_to(HERE)}")


def run(name: str, cfg: Config, widths: list, epochs: int, lr: float, batch: int,
        seeds=(0, 1, 2)) -> dict:
    """One arm of one sweep: the same setting trained from `seeds` initializations and
    shuffled orders, scored on the validation split every time."""
    per_seed, per_seed_datasets, curves, seconds, windows, params = [], [], [], 0.0, None, None
    for seed in seeds:
        seed_everything(seed)
        data = sf_data.build(cfg, verbose=False)
        sf_data.sets(data, ("train", "valid"))          # no test window is ever built here
        model = fitting.RateNet(widths)
        t0 = time.perf_counter()
        model, log = fitting.fit(model, data["train_set"], data, epochs=epochs, lr=lr,
                                 batch_size=batch, verbose=False)
        seconds += time.perf_counter() - t0
        s = fitting.correlation(model, data, "valid")
        per_seed.append(s["mean_over_neurons"])
        per_seed_datasets.append(s["mean_over_datasets"])
        curves.append([r["valid_corr"] for r in log["log"] if "valid_corr" in r])
        best = max((r["epoch"] for r in log["log"] if r.get("best")), default=0)
        windows, params = len(data["train_set"]), model.n_parameters
        print(f"     seed {seed}: valid {per_seed[-1]:.3f} at epoch {best:>3}/{epochs} "
              f"[{time.perf_counter() - t0:.0f} s]")
    out = {"name": name, "data": asdict(cfg), "widths": list(widths), "epochs": epochs,
           "lr": lr, "batch_size": batch, "seeds": list(seeds),
           "corr": round(float(np.mean(per_seed)), 4),
           "corr_spread": round(float(np.std(per_seed)), 4),
           "corr_per_seed": [round(v, 4) for v in per_seed],
           "corr_datasets": round(float(np.mean(per_seed_datasets)), 4),
           "seconds": round(seconds, 1), "windows_train": windows, "n_parameters": params,
           "curve": [round(float(np.mean([c[i] for c in curves])), 4)
                     for i in range(min(len(c) for c in curves))]}
    print(f"  {name:<36} valid {out['corr']:.3f} +- {out['corr_spread']:.3f} over "
          f"{len(seeds)} seeds   {windows:>5} windows   {seconds:>6.0f} s")
    return out


TIE = 0.005                                            # a gap this small is not a measurement
ABLATION = ("standardize none", "target scale")        # measured, not candidates
COSTED = ("lr", "batch")                               # sweeps whose arms differ only in price
KERNEL_SECONDS = 2.3        # the slowest calcium half-decay the audit measured, over 174 neurons
RF_SAMPLES = {3: 44, 4: 92, 5: 188}   # receptive field of part1.UNet(dim=1), in input samples,
                                      # measured per level count by pushing an impulse through it


def rf_seconds(widths, decimate: int) -> float:
    """How much time one output sample of a `widths` network actually sees, on a grid of
    `decimate` released samples. The audit measured the calcium kernel needing 0.4-2.3 s
    to fall to half its peak, so a field of view under 2.3 s cannot see a whole decay, and
    one over the window length is mostly zero padding."""
    return RF_SAMPLES[len(widths)] * decimate / sf_data.FS_IN


def pick(state: dict, sweep: str) -> dict:
    """The arm a sweep chooses, under one rule applied to every sweep alike.

    Highest validation correlation, with **ablations excluded** (the arms that turn the
    specification's own choices off to measure what they are worth - they are
    measurements, not candidates, and "no standardization" scoring within noise of the
    default is not a reason to drop it). Then, among the arms within `TIE` of the best -
    a gap the three seeds cannot resolve, since one arm's seed sd is ~0.003 while the 18
    validation neurons are worth +-0.045, so ranking them further would be ranking noise:

    * **lr and batch**: the arm that costs least, in measured seconds. A 2x larger batch
      for +0.001 is not a result worth 2x the wall clock.
    * **depth**: not cost at all but physics. The arm whose receptive field covers the
      audit's slowest calcium half-decay (2.3 s) without exceeding the window it is fed -
      a field of view shorter than the kernel cannot see a whole decay, and one longer
      than the window is mostly zero padding. Three depths scoring 0.628 / 0.630 / 0.631
      is a tie the metric cannot break; the kernel time can, and it is a number measured
      from the data before any network was trained.
    * **the data sweep**: the best score outright, because cost differs by construction
      there - the window size *is* the variable, so seconds do not measure a price paid
      for an advantage.

    RESULTS.md reports what each branch of the rule changed.
    """
    arms = [a for a in state["sweeps"][sweep] if not a["name"].startswith(ABLATION)]
    top = max(a["corr"] for a in arms)
    near = [a for a in arms if a["corr"] >= top - TIE]
    if sweep == "depth" and len(near) > 1:
        d, window_s = near[0]["data"]["decimate"], near[0]["data"]["window"] * \
            near[0]["data"]["decimate"] / sf_data.FS_IN
        fits = [a for a in near if KERNEL_SECONDS <= rf_seconds(a["widths"], d) <= window_s]
        if fits:
            chosen = max(fits, key=lambda a: a["corr"])
            print(f"   depth: {len(chosen['widths'])} levels sees "
                  f"{rf_seconds(chosen['widths'], d):.1f} s, which covers the audit's slowest "
                  f"kernel ({KERNEL_SECONDS} s) inside the {window_s:.1f} s window; the "
                  f"candidates that tie within {TIE} see "
                  + ", ".join(f"{rf_seconds(a['widths'], d):.1f} s" for a in near)
                  + f", and only these fit: {', '.join(str(len(a['widths'])) for a in fits)}")
            return chosen
        print(f"   depth: no candidate within {TIE} has a field of view inside the window; "
              f"taking the best score and saying so")
    if sweep in COSTED and len(near) > 1:
        return min(near, key=lambda a: a["seconds"])
    return max(near, key=lambda a: a["corr"])


def rechoose(state: dict) -> None:
    """Re-derive `chosen` from the arms already recorded, using pick(). Nothing is
    retrained - it re-reads measurements that exist - so this is what to run after
    `--refine` if the argmax and the rule disagree. The epoch count stays the one read off
    the long curve; if that curve ran at a depth the rule did not pick, this says so out
    loud, and stage 3's confirmation run then gives the epoch count an independent read at
    the depth that actually ships."""
    chosen, said = state["chosen"], []
    for sweep, field in (("lr", "lr"), ("depth", "widths"), ("batch", "batch_size")):
        if not state["sweeps"].get(sweep):
            continue
        arm = pick(state, sweep)
        if chosen.get(field) != arm[field]:
            said.append(f"{field}: {chosen[field]} -> {arm[field]} ({arm['name']}, "
                        f"{arm['corr']:.3f} vs best {max(a['corr'] for a in state['sweeps'][sweep]):.3f})")
        chosen[field] = arm[field]
    curve = (state["sweeps"].get("epochs") or [{}])[0]
    if curve.get("widths") and curve["widths"] != chosen["widths"]:
        said.append(f"the {curve['epochs']}-epoch curve ran at widths={curve['widths']}, not "
                    f"the shipped {chosen['widths']} - stage 3 re-reads the epoch at the "
                    f"shipped depth")
    print("rechoose: " + ("; ".join(said) if said else "the rule agrees with the recorded argmax"))


# --------------------------------------------------------------------------- #
def stage_one(state: dict, epochs: int, seeds) -> None:
    print(f"\nstage 1 - the data. {epochs} epochs x {len(seeds)} seeds per arm; every arm "
          f"shares widths={WIDTHS}, lr 3e-3, batch 32.")
    arms = [run(f"decimate {d} ({100 // d} Hz, window {round(20.5 * 100 / d)})",
                Config(decimate=d, window=round(20.5 * 100 / d)), WIDTHS, epochs, 3e-3, 32, seeds)
            for d in (1, 2, 4)]
    arms += [run(f"window {w} samples ({w / 25:.0f} s)", Config(window=w), WIDTHS, epochs,
                 3e-3, 32, seeds) for w in (128, 256, 1024)]
    arms += [run("standardize none", Config(standardize="none"), WIDTHS, epochs, 3e-3, 32, seeds),
             run("target scale global", Config(scale="global"), WIDTHS, epochs, 3e-3, 32, seeds)]
    state["sweeps"]["data"] = arms
    state["curves"] = [a for a in arms if a["name"].startswith(("decimate 4", "decimate 1",
                                                                "target scale"))]
    winner = pick(state, "data")
    state["chosen"] = {"data": winner["data"], "widths": WIDTHS, "lr": 3e-3, "batch_size": 32,
                       "epochs": epochs, "baseline_sigma": 5, "seed": 0}
    print(f"\n   stage 1 winner: {winner['name']} at {winner['corr']:.3f}")
    save(state)
    figure(state)


def stage_two(state: dict, epochs: int, seeds, long: int = LONG) -> None:
    chosen = state["chosen"]
    cfg = Config(**chosen["data"])
    print(f"\nstage 2 - the optimizer on {cfg.key()}. {epochs} epochs x {len(seeds)} seeds per arm.")
    lr_arms = [run(f"lr {lr:g}", cfg, chosen["widths"], epochs, lr, chosen["batch_size"], seeds)
               for lr in (1e-3, 3e-3, 1e-2)]
    state["sweeps"]["lr"] = lr_arms
    lr = pick(state, "lr")["lr"]

    depth_arms = [run(f"{k} levels {fitting.widths_for(k)}", cfg, fitting.widths_for(k), epochs,
                      lr, chosen["batch_size"], seeds) for k in (3, 4, 5)]
    state["sweeps"]["depth"] = depth_arms
    widths = pick(state, "depth")["widths"]

    batch_arms = [run(f"batch {b}", cfg, widths, epochs, lr, b, seeds) for b in (16, 32, 64)]
    state["sweeps"]["batch"] = batch_arms
    batch = pick(state, "batch")["batch_size"]

    print(f"\n   the long curve at the winner (lr {lr:g}, {widths}, batch {batch}): "
          f"{long} epochs x 1 seed")
    curve = run(f"long curve, {len(widths)} levels", cfg, widths, long, lr, batch, seeds=(0,))
    state["sweeps"]["epochs"] = [curve]
    state["curves"] = [curve] + state.get("curves", [])[:2]
    best = int(np.argmax(curve["curve"])) + 1
    print(f"   the validation correlation peaks at epoch {best} of {long}; the training loss "
          f"is still falling there (its curve is in the json) - the gap between the two is "
          f"why the epoch is chosen on the correlation")
    chosen.update({"lr": lr, "widths": widths, "batch_size": batch,
                   "epochs": int(min(long, max(24, round(best * 1.1))))})
    state["chosen"] = chosen
    save(state)
    figure(state)


def stage_three(state: dict, seeds) -> None:
    chosen = state["chosen"]
    cfg = Config(**chosen["data"])
    print(f"\nstage 3 - confirmation at {chosen['epochs']} epochs x {len(seeds)} seeds.")
    runs = [run("chosen", cfg, chosen["widths"], chosen["epochs"], chosen["lr"],
                chosen["batch_size"], seeds)]
    pool = state["sweeps"]["lr"] + state["sweeps"]["depth"] + state["sweeps"]["batch"]
    runner = max((a for a in pool if a["name"] != "chosen"), key=lambda a: a["corr"])
    print(f"   runner-up from stage 2: {runner['name']} at {runner['corr']:.3f}")
    runs.append(run(f"runner-up: {runner['name']}", Config(**runner["data"]), runner["widths"],
                    chosen["epochs"], runner["lr"], runner["batch_size"], seeds))
    twin = copy.deepcopy(cfg)
    twin.drop_duplicate = False
    runs.append(run("duplicate (9,3) back in train", twin, chosen["widths"], chosen["epochs"],
                    chosen["lr"], chosen["batch_size"], seeds))

    # The confound stage 1 left standing, measured at the length the model trains for.
    # Shorter windows win on more than context: at a fixed epoch budget 4x as many
    # windows is 4x as many gradient steps, so "5 s per window" and "4x more updates"
    # are the same experiment unless they are separated. Here they are: raising the batch
    # by the window ratio puts the short window back on the same number of updates per
    # epoch as a 512-sample one, and the 512 arm is re-run at this length to compare with.
    wide = copy.deepcopy(cfg)
    wide.window = 512
    mult = max(1, 512 // cfg.window)          # the window-size ratio, i.e. the step-ratio fix
    print(f"\n   context or updates? the short-window winner re-run at batch x{mult} (the same "
          f"steps per epoch as a 512-sample window) against a 512-sample window:")
    runs.append(run(f"window {cfg.window}, batch {chosen['batch_size'] * mult}", cfg,
                    chosen["widths"], chosen["epochs"], chosen["lr"],
                    chosen["batch_size"] * mult, seeds))
    runs.append(run("window 512, batch as chosen", wide, chosen["widths"], chosen["epochs"],
                    chosen["lr"], chosen["batch_size"], seeds))
    state["sweeps"]["confirm"] = runs
    state["chosen"]["baseline_sigma"] = _baseline_sigma()
    save(state)
    spread = max(r["corr"] for r in runs) - min(r["corr"] for r in runs)
    print(f"\n   the full-length runs span {spread:.3f}; the +-0.045 interval on 18 neurons means "
          f"this stage rules settings out rather than ranking them")


def _baseline_sigma() -> int:
    import baseline
    return baseline.tune(sf_data.build(sf_data.configured(), verbose=False), verbose=False)


# --------------------------------------------------------------------------- #
REF = {"data": "decimate 4", "lr": "lr 0.003", "depth": "3 levels", "batch": "batch 32",
       "confirm": "chosen"}


def figure(state: dict) -> None:
    """One panel per sweep: the arms' validation correlation, the sd over seeds as a
    whisker, and each value printed with its difference from the specification's default
    arm. The epoch curves go in their own figure, `curves_figure`. Same palette as the
    audit figures in ../task3_audit_figs/.

    The bars start at zero, because a correlation of zero means something and a
    truncated axis would turn a +0.010 into a visual landslide; the differences are the
    finding, so they are printed next to the values instead. The whiskers are the true
    seed spread, which on most arms is too small to see - that is the result the three
    seeds were bought for, and the reason these arm-to-arm gaps are believed while an
    absolute validation score of 0.628 is not (the 18 neurons behind it are worth +-0.045).
    """
    import matplotlib.pyplot as plt

    import figs
    figs.style()
    blue, orange, ink, muted = figs.BLUE, figs.ORANGE, figs.INK, figs.MUTED
    sweeps = [(k, v) for k, v in state["sweeps"].items() if v and k != "epochs"]
    if not sweeps:
        return
    n = len(sweeps)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 2.9), squeeze=False)
    for ax, (name, arms) in zip(axes[0], sweeps):
        ref = next((a["corr"] for a in arms if a["name"].startswith(REF.get(name, ""))), None)
        values = [a["corr"] for a in arms]
        y = np.arange(len(arms))[::-1]
        ax.barh(y, values, color=[orange if v == max(values) else blue for v in values],
                height=0.62)
        for k, a in enumerate(arms):
            ax.errorbar(a["corr"], y[k], xerr=a.get("corr_spread", 0.0), color=ink, lw=0.9,
                        capsize=2.5, zorder=5)
            delta = "" if ref is None else f"  {a['corr'] - ref:+.3f}"
            ax.text(a["corr"] + 0.02, y[k], f"{a['corr']:.3f}{delta}", va="center",
                    fontsize=6.5, color=ink)
        ax.set_yticks(y)
        ax.set_yticklabels([a["name"] for a in arms], fontsize=7)
        ax.set_xlim(0, max(values) * 1.30 + 0.04)
        ax.axvline(0, color=muted, lw=0.8)
        ax.set_title(name, loc="left", fontsize=8.5)
        ax.set_xlabel(f"validation correlation, mean of {len(arms[0]['seeds'])} seed"
                      f"{'s' if len(arms[0]['seeds']) > 1 else ''}\n"
                      "(whisker: sd over seeds; text: value, difference from the spec "
                      "default)" if ax is axes[0][0] else "")
    fig.tight_layout()
    figs.save(fig, "validation_choices.png")
    curves_figure(state)


def curves_figure(state: dict) -> None:
    """The validation-correlation-versus-epoch curves, on their own: the short arms from
    the last sweep and, once `--refine` has run, the long curve the epoch count was read
    off. The dashed line on the long one is the epoch budget the final run gets: it sits
    past the peak, because `stage_two` floors the budget at 24 epochs, and that is only
    harmless while `fit` keeps the best epoch rather than the last one. It has a price,
    stated in RESULTS.md: keeping the best of 24 looks at 18 neurons makes the selected
    validation score optimistic by about the epoch-to-epoch wobble of this curve. Each
    curve is the mean over that arm's seeds and the ring marks *its* maximum, which is not
    quite the mean of the per-seed maxima that `run` reports and selects on."""
    import matplotlib.pyplot as plt

    import figs
    curves = state.get("curves", [])
    long_ = [a for a in curves if a["name"].startswith("long curve")]
    short = [a for a in curves if not a["name"].startswith("long curve")]
    LONG_TITLE = "the chosen setting, trained on for 90 epochs"
    panels = [(arms, title, loc) for arms, title, loc in (
        (short[:3], "the data arms of stage 1, scored every epoch", "lower right"),
        (long_[:1], LONG_TITLE, "upper right")) if arms]
    if not panels:
        return
    figs.style()
    fig, axes = plt.subplots(1, len(panels), figsize=(3.7 * len(panels), 2.8), squeeze=False)
    for ax, (arms, title, loc) in zip(axes[0], panels):
        for arm, color in zip(arms, figs.SERIES):
            ep = range(1, len(arm["curve"]) + 1)
            ax.plot(ep, arm["curve"], color=color, lw=1.5, marker="o", ms=2.5,
                    label=f"{arm['name']} ({len(arm['curve'])} ep)")
            peak = int(np.argmax(arm["curve"])) + 1
            ax.plot(peak, arm["curve"][peak - 1], "o", ms=6, mfc="none", mec=color, mew=1.2)
        if title == LONG_TITLE and "epochs" in state.get("chosen", {}):
            figs.annotate_peak(ax, state["chosen"]["epochs"], max(arms[0]["curve"]),
                               f"the budget: {state['chosen']['epochs']} epochs", figs.MUTED)
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation correlation")
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.legend(fontsize=5.5, loc=loc, frameon=True, facecolor=figs.SURFACE,
                  edgecolor=figs.GREY, framealpha=0.92)
    fig.tight_layout()
    figs.save(fig, "validation_curves.png")


def start() -> dict:
    """What every entry point - this script, and `run_all.py` driving the three stages
    in order - does before any training: line-buffered stdout so a `| tee` log shows
    progress live, the one rule this file exists to keep, and the thread cap every
    timing quoted here was measured under."""
    sys.stdout.reconfigure(line_buffering=True)
    data = sf_data.build(Config(), verbose=False)
    assert not any(n in data["split"]["test"] for n in data["mu"]), "a test neuron's labels leaked"
    del data
    torch.set_num_threads(6)
    return load()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refine", action="store_true", help="stage 2: lr, depth, batch, epochs")
    ap.add_argument("--confirm", action="store_true", help="stage 3: full-length confirmation")
    ap.add_argument("--epochs", type=int, default=SHORT, help="epochs per arm in stages 1-2")
    ap.add_argument("--seeds", type=int, default=3, help="repeats per arm (1 is fast and noisy)")
    ap.add_argument("--rechoose", action="store_true",
                    help="re-derive the choice from the recorded arms under pick()'s rule; "
                         "trains nothing, changes results/choices.json only")
    args = ap.parse_args()

    if args.rechoose:
        state = load()
        if not state.get("chosen"):
            raise SystemExit("no results/choices.json to re-choose from - run stage 1 first")
        rechoose(state)
        save(state)
        return

    state, seeds = start(), tuple(range(args.seeds))
    t0 = time.perf_counter()
    if args.confirm:
        stage_three(state, seeds)
    elif args.refine:
        stage_two(state, args.epochs, seeds)
    else:
        stage_one(state, args.epochs, seeds)
    print(f"\nstage took {(time.perf_counter() - t0) / 60:.1f} min on {part1.device}")


if __name__ == "__main__":
    main()
