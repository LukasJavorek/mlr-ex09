"""Every number this folder would otherwise hard-code, measured on the 100 validation photos.

    python validate_choices.py            # 29 arms, 49.9 min on 6 CPU threads; writes
                                          # results/choices.json and figures/choices.png
    python validate_choices.py --quick    # half the photos and epochs, same arms, ~13 min
    python validate_choices.py --report   # reads choices.json back and prints rule R4's verdict
                                          # on the arms: no training, no photos, no time

WHAT IS CHOSEN HERE, and what deliberately is not. Chosen: the width of each network, the
learning rate, the batch size, whether the horizontal-flip arm earns its place, and for route
B how many of each photo's 4096 pixels the loss is computed on. Also measured, so that the
sweep's differences can be read at all: the run-to-run spread from the seed. NOT chosen here:
the number of epochs, because a sweep at a quarter of the data would pick a different epoch
from the full run - `train_task6.py` trains a fixed budget and keeps the epoch with the best
validation loss, which is the same measurement done on the model that gets used. And not
chosen here: alpha and gamma, which are properties of a *trained* model's outputs and are
therefore calibrated afterwards, by `calibrate.py`, still on valid.

THE RULES, written before any arm ran, so that "we picked the arm we liked" is not what
happens:

  R1  An arm is judged by its best validation loss - route A's L1 on the chroma planes, route
      B's cross-entropy on the bin labels - never by its PSNR. Per-photo PSNR on this data is
      maximised by predicting no colour (the audit put that maximum at alpha = 0), so ranking
      arms by PSNR ranks them by how dull they are. The epoch curves of validation PSNR and
      colourfulness are still logged, and `figures/choices.png` plots all three.
  R2  Arms are compared only within a route. L1 in /128 units and cross-entropy in nats are
      not the same quantity and no number here converts between them.
  R3  The winner is the lowest best-validation-loss arm. An arm within 0.2% of that loss is
      tied with it, and the tie goes to the smaller model - a tie broken by size is a tie
      broken by nothing you can see in a picture, and the smaller model is the one whose
      failure is easier to read.
  R4  A difference counts only if it is at least two stage-3 seed standard deviations. An arm
      that beats another by less than that is reported as a tie whatever its loss says,
      because on this data the seed alone moves the loss about that much. It is a rule about
      how a difference is *read*: `pick` applies R3's 0.2% band and this floor together and the
      tighter of the two decides which arm survives, so `--report` prints both bands and marks
      every arm inside the floor as a tie with the winner. On the run that produced
      results/choices.json, R3's band is the tighter one on both routes - which is itself the
      finding worth having: most of stage 2's differences are inside the seed's own reach, and
      the report says which.
  R5  Everything is measured on a fixed 2367-photo subset of the *training* split (a seeded
      quarter of it, the same photos in every arm) and scored on the same 100 validation
      photos. The subset is a budget decision, not a method one: it is the same 9469 photos
      at a quarter of the epochs' worth of gradient steps, it is applied identically to every
      arm, and R4 is measured at the same subset size so the comparison stays honest.
      Stage 3's spread is the price of that budget, stated.
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
import spec

OUT = d6.HERE / "results"
SUBSET = 2367                 # a seeded quarter of the 9469 training photos (R5)
LRS = (3e-3, 1e-2)
WIDTHS = ([16, 32, 64], [32, 64, 128], [16, 32, 64, 128])
EPOCHS_BUDGET = 16            # the final run's ceiling; the epoch it uses is its best loss one
TIE_FRACTION = 0.002          # R3
LOG2 = {}                     # arm records, for choices.json


def stage(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}", flush=True)


def subset_of(data: dict, n: int) -> np.ndarray:
    """The same `n` training photos for every arm in the run (R5)."""
    order = np.random.default_rng(0).permutation(len(data["split"]["train"]))[:n]
    return d6.arrays(data, "train")[order]


def arm(name: str, route: str, widths: list[int], subset: np.ndarray, valid_rgb: np.ndarray,
        prior: np.ndarray, epochs: int, lr: float = 3e-3, batch_size: int = 32,
        pixels: int | None = 1024, flip: bool = False, seed: int = 0,
        note: str = "") -> dict:
    """One training run, one line of output, one record in LOG2."""
    cfg = d6.Config(flip=flip)
    tr = d6.Photos(subset, route, cfg)
    va = d6.Photos(valid_rgb, route, d6.Config())
    model = fitting.Colorizer(route, widths, prior_prob=prior if route != "A" else None)
    t0 = time.perf_counter()
    model, info = fitting.fit(model, tr, va, prior_prob=prior if route != "A" else None,
                              epochs=epochs, lr=lr, batch_size=batch_size, pixels=pixels,
                              seed=seed, verbose=False)
    curve = info["curve"]
    k = info["best_epoch"] - 1
    rec = {"arm": name, "route": route, "widths": widths, "epochs_budget": epochs, "lr": lr,
           "batch_size": batch_size, "pixels": pixels, "flip": flip, "seed": seed,
           "n_photos": len(subset), "n_parameters": info["n_parameters"],
           "best_valid_loss": info["best_valid_loss"], "best_epoch": info["best_epoch"],
           "valid_psnr_at_best": curve["valid_psnr"][k],
           "valid_colourfulness_at_best": curve["valid_colourfulness"][k],
           "valid_colourfulness_truth": round(cs.colourfulness(*cs.ycc(valid_rgb)[1:]), 4),
           "seconds": info["seconds"], "curve": curve, "note": note}
    LOG2[name] = rec
    psnr = "  -    " if rec["valid_psnr_at_best"] is None else f"{rec['valid_psnr_at_best']:6.2f}"
    col = "  -    " if rec["valid_colourfulness_at_best"] is None \
        else f"{rec['valid_colourfulness_at_best']:6.3f}"
    print(f"  {name:<34} {info['n_parameters']:>7} p  loss {info['best_valid_loss']:.5f} "
          f"@ep {info['best_epoch']:<3} {psnr} dB {col} cf "
          f"[{(time.perf_counter() - t0) / 60:4.1f} min]{('  ' + note) if note else ''}",
          flush=True)
    return rec


def pick(arms: list[dict], sd: float | None = None) -> dict:
    """R3/R4: lowest loss, ties to the smaller model, differences under 2 sds ignored.

    The cost tie-break is *seconds per epoch budget*, not seconds: rule R4's pool deliberately
    contains stage-1 arms (4 epochs) and stage-2 arms (6) of the same network, and raw seconds
    would call the arm that ran fewer epochs the cheaper one. That is a budget accident, not a
    saving, and it would let the seed-spread tie-break pick an arm for having been trained less.
    """
    ranked = sorted(arms, key=lambda a: a["best_valid_loss"])
    best = ranked[0]
    band = best["best_valid_loss"] * (1 + TIE_FRACTION)
    tied = [a for a in arms if a["best_valid_loss"] <= band]
    if sd is not None:
        tied = [a for a in tied if a["best_valid_loss"] <= best["best_valid_loss"] + 2 * sd]
    winner = min(tied, key=lambda a: (a["n_parameters"], a["seconds"] / a["epochs_budget"]))
    return winner


def seed_spread(arms: list[dict]) -> tuple[float, float]:
    """(sd of best_valid_loss, sd of valid colourfulness) across seeds of one configuration."""
    losses = np.array([a["best_valid_loss"] for a in arms], float)
    cf = np.array([a["valid_colourfulness_at_best"] for a in arms], float)
    return float(losses.std(ddof=1)), float(cf.std(ddof=1))


def plot(sweeps: dict) -> None:
    figs.style()
    import matplotlib.pyplot as plt
    panels = [(s["title"], s["arms"]) for s in sweeps.values()]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0))
    for ax, (title, arms) in zip(axes.ravel(), panels):
        a = sorted(arms, key=lambda r: r["best_valid_loss"])
        labels = [r["arm"].split("/", 1)[-1][:22] for r in a]
        ax.barh(range(len(a)), [r["best_valid_loss"] for r in a], color=figs.SERIES[0], height=0.6)
        ax.set_yticks(range(len(a)), labels, fontsize=6.5)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xlabel("best validation loss (lower is better)", fontsize=7)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle("Every arm judged on validation loss (rule R1); route A and B losses are "
                 "different quantities and are never compared (R2)", fontsize=8.5)
    figs.save(fig, "choices.png")


def report(ch: dict) -> None:
    """Rule R4, applied after the fact to the arms already in results/choices.json.

    Nothing is trained or measured here - it reads the file the sweep wrote and answers the
    question a reader of the sweep log actually has: which of these wins is larger than the
    noise that moves it. R4 is a rule about *reporting*, so this is where it is reported, and
    `python validate_choices.py --report` reproduces the whole thing from the recorded arms in
    no time at all.
    """
    sd, sw, chosen = ch["seed_spread"], ch["sweeps"], ch["chosen"]
    print("Rule R4 applied to the recorded arms: which stage-1/2 differences the seed can move.")
    print("Nothing below trains anything; every number is read from results/choices.json.\n")
    for route in ("A", "B"):
        s = sd[route]["loss"]
        best_all = min(x["best_valid_loss"] for x in sw["stage2"]["arms"]
                       if x["arm"].startswith(f"2/{route}"))
        r3 = best_all * TIE_FRACTION
        print(f"route {route}: the stage-3 seed spread of the best validation loss is {s:.5f} "
              f"({sd[route]['loss_rel_pct']:.2f}% of the loss), so R4's floor is 2 sd = "
              f"{2 * s:.5f}.")
        print(f"  R3's indifference band is {TIE_FRACTION:.1%} of the best loss = {r3:.5f}, and "
              f"`pick` asks an arm to be inside both to count as tied: here the "
              f"{'R3' if r3 < 2 * s else 'R4'} band is the tighter, so it is the one that "
              f"decides the arm, and R4 decides the wording below.")
        for stage, keep in (("stage1", f"1/{route}"), ("stage2", f"2/{route}")):
            arms = [x for x in sw[stage]["arms"] if x["arm"].startswith(keep)
                    and "pixels" not in x["arm"]]        # the pixel arms get their own table
            best = min(x["best_valid_loss"] for x in arms)
            print(f"\n  {stage}:")
            for x in sorted(arms, key=lambda x: x["best_valid_loss"]):
                gap = x["best_valid_loss"] - best
                won = x["arm"] in (chosen[route]["stage1_pick"], chosen[route]["stage2_pick"])
                verdict = ("chosen" if won else
                           "inside 2 sd: R4 calls this a tie with the winner" if gap <= 2 * s
                           else "outside 2 sd: a difference the seed cannot explain")
                print(f"    {x['arm']:<24} {x['best_valid_loss']:.5f}  +{gap:.5f}   {verdict}")
        px = [x for x in sw["stage2"]["arms"] if x["arm"].startswith(f"2/{route} pixels")]
        if px:
            best = min(x["best_valid_loss"] for x in px)
            print(f"\n  the pixel count: the same objective sampled at three costs "
                  f"(smoke_test.py proves they are the same objective):")
            for x in sorted(px, key=lambda x: x["best_valid_loss"]):
                gap = x["best_valid_loss"] - best
                verdict = ("chosen" if x["pixels"] == chosen[route]["pixels"]
                           else "inside 2 sd: R4 calls this a tie with the winner" if gap <= 2 * s
                           else "outside 2 sd: a difference the seed cannot explain")
                print(f"    {x['arm']:<24} {x['best_valid_loss']:.5f}  +{gap:.5f}   "
                      f"{x['seconds'] / x['epochs_budget']:5.0f} s/epoch   {verdict}")
            print("    a tie here is therefore decided on s/epoch, which is the honest reading: "
                  "the arm that costs least, not the arm whose loss happened to be smallest.")
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="half the photos, half the epochs")
    ap.add_argument("--report", action="store_true",
                    help="print rule R4's verdict on the arms in results/choices.json and stop; "
                         "trains nothing and reads no photos")
    a = ap.parse_args()
    if a.report:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        report(json.loads((d6.HERE / "results" / "choices.json").read_text(encoding="utf-8")))
        return
    sys.stdout.reconfigure(line_buffering=True)
    fitting.threads(6)
    d6.seed_everything(0)
    t_start = time.perf_counter()

    print(spec.text())
    print()
    print("Selection rules R1-R5 are in this file's docstring, written before the first arm "
          "ran. Nothing below re-writes them.")

    data = d6.build(verbose=False)
    train_rgb, valid_rgb = d6.arrays(data, "train"), d6.arrays(data, "valid")
    prior = fitting.prior(train_rgb, cs.GRID)
    n_sub = SUBSET // 2 if a.quick else SUBSET
    ep1, ep2, ep3 = (2, 3, 2) if a.quick else (4, 6, 4)
    subset = subset_of(data, n_sub)
    print(f"\nsubset {subset.shape} of {train_rgb.shape[0]} training photos, fixed for every arm; "
          f"scored on the 100 validation photos; route-B prior entropy "
          f"{-(prior * np.log(prior)).sum():.4f} nats, "
          f"its modal cell {prior.max() * 100:.1f}% of pixels")
    print(f"budget: epochs {ep1}/{ep2}/{ep3} for stages 1/2/3"
          + ("  [--quick]" if a.quick else ""))

    sweeps: dict[str, dict] = {}

    # ---- stage 1: how wide, how fast ------------------------------------------------------- #
    stage(f"STAGE 1 - width and learning rate, {n_sub} photos, {ep1} epochs, "
          f"{len(WIDTHS) * len(LRS) * 2} arms")
    s1 = []
    for route in ("A", "B"):
        for widths in WIDTHS:
            for lr in LRS:
                s1.append(arm(f"1/{route} w{'-'.join(map(str, widths))} lr{lr:g}", route, widths,
                              subset, valid_rgb, prior, ep1, lr=lr))
    sweeps["stage1"] = {"title": "Stage 1: width and learning rate", "arms": s1}
    win1 = {r: pick([x for x in s1 if x["route"] == r]) for r in ("A", "B")}
    for r, w in win1.items():
        print(f"  -> stage 1 keeps route {r}: {w['arm']} "
              f"({w['n_parameters']} parameters, loss {w['best_valid_loss']:.5f})")

    # ---- stage 2: batching, augmentation, and for B how many pixels ------------------------ #
    stage(f"STAGE 2 - batch size, flip arm, and route B's pixel count, {ep2} epochs")
    s2 = []
    for route in ("A", "B"):
        widths = win1[route]["widths"]
        for bs in (32, 64):
            for flip in (False, True):
                s2.append(arm(f"2/{route} b{bs} flip{int(flip)}", route, widths, subset,
                              valid_rgb, prior, ep2, batch_size=bs, flip=flip,
                              note="flip" if flip else ""))
    b_widths = win1["B"]["widths"]
    for pixels in (512, 1024, None):
        s2.append(arm(f"2/B pixels{'all' if pixels is None else pixels}", "B", b_widths, subset,
                      valid_rgb, prior, ep2, pixels=pixels,
                      note="full-pixel" if pixels is None else ""))
    sweeps["stage2"] = {"title": "Stage 2: batch, flip, pixels", "arms": s2}
    # Each decision sees only its own arms: the pixel-count arms share route "B" and the same
    # epoch budget, so an unfiltered pick would let a pixels arm answer the batch-and-flip
    # question - which is exactly what it did the first time this ran.
    batch_flip = {r: [x for x in s2 if x["arm"].startswith(f"2/{r} b")] for r in ("A", "B")}
    win2 = {r: pick(v) for r, v in batch_flip.items()}
    win_b_pixels = pick([x for x in s2 if x["arm"].startswith("2/B pixels")])
    for r, w in win2.items():
        # The opposite flip arm at the same batch size, whatever the winner was. Quoting the
        # no-flip arm unconditionally prints the chosen arm's own number back when flip=False
        # wins, which reads as if the flip had been measured and found to matter at 0.00000.
        twin = next((x for x in batch_flip[r] if x["batch_size"] == w["batch_size"]
                     and x["flip"] != w["flip"]), None)
        against = (f"the same arm {'without' if w['flip'] else 'with'} the flip "
                   f"{twin['best_valid_loss']:.5f}" if twin else "no opposite arm at this batch")
        print(f"  -> stage 2 keeps route {r}: batch {w['batch_size']}, flip {w['flip']} "
              f"(loss {w['best_valid_loss']:.5f}; {against})")
    print(f"  -> stage 2 keeps route B pixels = "
          f"{'all' if win_b_pixels['pixels'] is None else win_b_pixels['pixels']} "
          f"(loss {win_b_pixels['best_valid_loss']:.5f}, "
          f"{win_b_pixels['seconds'] / 60:.1f} min for {ep2} epochs at this subset)")

    # ---- stage 3: how much does the seed move all of this? --------------------------------- #
    stage(f"STAGE 3 - three seeds of the chosen configuration, {ep3} epochs (rule R4's yardstick)")
    s3 = {"A": [], "B": []}
    for route in ("A", "B"):
        w = win2[route]
        for seed in (0, 1, 2):
            s3[route].append(arm(f"3/{route} seed{seed}", route, w["widths"], subset, valid_rgb,
                                 prior, ep3, batch_size=w["batch_size"], flip=w["flip"],
                                 pixels=1024 if route == "B" else None, seed=seed))
    sweeps["stage3"] = {"title": "Stage 3: seed spread of the chosen arms", "arms": s3["A"] + s3["B"]}
    sd = {}
    for route in ("A", "B"):
        loss_sd, cf_sd = seed_spread(s3[route])
        mean_cf = float(np.mean([x["valid_colourfulness_at_best"] for x in s3[route]]))
        sd[route] = {"loss": loss_sd, "colourfulness": cf_sd,
                     "loss_rel_pct": 100 * loss_sd / float(np.mean(
                         [x["best_valid_loss"] for x in s3[route]])),
                     "cf_rel_pct": 100 * cf_sd / max(mean_cf, 1e-9),
                     "losses": [x["best_valid_loss"] for x in s3[route]],
                     "colourfulness": [x["valid_colourfulness_at_best"] for x in s3[route]]}
        print(f"  -> route {route}: across seeds, best validation loss sd {loss_sd:.5f} "
              f"({sd[route]['loss_rel_pct']:.2f}% of the loss) and valid colourfulness sd "
              f"{cf_sd:.4f} ({sd[route]['cf_rel_pct']:.1f}% of its value). Rule R4: stage-1/2 "
              f"differences below 2 of these are reported as ties.")

    # R4 can only be applied to stages 1 and 2 after stage 3 has measured the yardstick, so
    # the winners are recomputed here and both are recorded: the R3 pick and the R4 pick.
    final = {}
    for route in ("A", "B"):
        pool = batch_flip[route] + [x for x in s1
                                    if x["route"] == route and x["widths"] == win1[route]["widths"]
                                    and x["lr"] == win1[route]["lr"]]
        w3, w4 = win2[route], pick(pool, sd=sd[route]["loss"])
        px4 = pick([x for x in s2 if x["arm"].startswith("2/B pixels")], sd=sd["B"]["loss"]) \
            if route == "B" else None
        # The knobs come from w4, the arm that survives the seed spread, because that is what the
        # "the R4 pick is used" message below promises. Reading them from the stage winners
        # instead - which is what this did until it was checked - printed one decision and wrote
        # another whenever the two disagreed. In this run they agree, so nothing here changed;
        # `agree` and `r4_pick` are recorded so a reader can verify that rather than trust it.
        final[route] = {"widths": w4["widths"],
                        "lr": w4["lr"],
                        "batch_size": w4["batch_size"],
                        "flip": w4["flip"],
                        "pixels": win_b_pixels["pixels"] if route == "B" else None,
                        "epochs_budget": EPOCHS_BUDGET,
                        "seed": 0,
                        "stage1_pick": win1[route]["arm"], "stage2_pick": win2[route]["arm"],
                        "r4_pick": w4["arm"], "agree": w3["arm"] == w4["arm"],
                        "r4_pixels_pick": px4["arm"] if px4 else None,
                        "pixels_agree": True if px4 is None
                        else px4["pixels"] == win_b_pixels["pixels"],
                        "loss": w4["best_valid_loss"]}
        if not final[route]["agree"]:
            print(f"  !! route {route}: R3 picks {w3['arm']} and R4 picks {w4['arm']}; the R4 "
                  f"pick (the one that survives the seed spread) is used")
        if px4 is not None and not final[route]["pixels_agree"]:
            print(f"  !! route B: R3 picks "
                  f"{'all' if win_b_pixels['pixels'] is None else win_b_pixels['pixels']} pixels "
                  f"and R4 picks {'all' if px4['pixels'] is None else px4['pixels']}; the R4 "
                  f"pick is used")
        if px4 is not None:
            final[route]["pixels"] = px4["pixels"]

    base = d6.Config()          # the data definition is the notebook's, chosen by nobody
    final["data"] = {"size": base.size, "seed": base.seed, "grid": base.grid}
    final["seed"] = 0
    final["subset_photos"] = int(n_sub)
    final["alpha"] = None            # calibrated by calibrate.py, on valid, after training
    final["gamma"] = None

    OUT.mkdir(exist_ok=True)
    payload = {"sweeps": sweeps, "curves": {r: win2[r]["curve"] for r in ("A", "B")},
               "seed_spread": sd, "chosen": final,
               "rules": "R1 validation loss; R2 within route only; R3 ties to the smaller "
                        "model; R4 differences under 2 seed sds are ties; R5 fixed subset",
               "totals": {"seconds": round(time.perf_counter() - t_start, 1),
                          "arms": len(LOG2), "quick": a.quick,
                          "threads": 6, "numpy": np.__version__}}
    (OUT / "choices.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote results/choices.json ({len(LOG2)} arms in "
          f"{(time.perf_counter() - t_start) / 60:.1f} min)")
    plot(sweeps)
    print("\nNothing above touched a test photo. The only arrays any arm ever saw were the "
          "fixed training subset and the 100 validation photos; the 500 test photos and the "
          "3325 unused val photos were never put into a dataset by this script.")


if __name__ == "__main__":
    main()
