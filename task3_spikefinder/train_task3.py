"""The final training run of Task 3: one network, on the settings validate_choices.py
measured on the validation split, with the test split untouched.

    python train_task3.py

Reads `results/choices.json` (run `python validate_choices.py` first, or edit
`sf_data.Config` defaults and `CHOSEN` below and delete the file to train the
specification as written). Writes `results/model.pt`, `results/train_log.json` and
`figures/training_curves.png`.

Which epoch's weights are kept is the interesting part of this script. `part1.train`
keeps the best validation *loss*; this keeps the best validation *correlation*, the
number the challenge scores, via `fitting.fit`. The two criteria need not agree, because
the loss cares about the level of the predicted rate and the score cannot see the level at
all. In the run this repository ships they nearly do - epoch 11 by correlation (0.635),
epoch 13 by validation loss - while the *training* loss keeps falling monotonically
(-0.46 to -1.39) long after the correlation peaked, which is the case that would fool a
loop selecting on loss: at epoch 24 the model is better by the loss and worse by 0.022 on
the metric. `figures/training_curves.png` prints all three curves for that reason. What is
saved to `model.pt` is the state of the best-correlation epoch, reloaded, and
`train_log.json` records which epoch that was, so nothing has to trust that the file
matches the number.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict

import figs
import fitting
import metrics
import part1
import sf_data
import spec
from sf_data import seed_everything
from validate_choices import CHOICES, RESULTS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)      # so `| tee results/train_log.txt` is live
    print(spec.text())
    if not CHOICES.exists():
        raise SystemExit("no results/choices.json - run `python validate_choices.py` first "
                         "(or copy choices.json.example)")
    chosen = json.loads(CHOICES.read_text(encoding="utf-8"))["chosen"]
    cfg, widths = sf_data.Config(**chosen["data"]), chosen["widths"]
    epochs, lr, batch, seed = chosen["epochs"], chosen["lr"], chosen["batch_size"], chosen["seed"]
    print(f"\nTRAINING  widths={widths}  lr={lr:g}  batch={batch}  epochs={epochs}  seed={seed}")
    print(f"          data: {cfg.key()}, window {cfg.window} = {cfg.seconds:.1f} s, "
          f"target scale {cfg.scale}, duplicate dropped: {cfg.drop_duplicate}")

    torch.set_num_threads(6)
    seed_everything(seed)
    data = sf_data.build(cfg)
    sf_data.sets(data)
    model = fitting.RateNet(widths).to(part1.device)
    print(f"          {model.n_parameters / 1e6:.2f} M parameters, "
          f"{len(data['train_set'])} training windows, "
          f"{len(data['split']['valid'])} validation neurons scored every epoch\n")

    t0 = time.perf_counter()
    model, log = fitting.fit(model, data["train_set"], data, epochs=epochs, lr=lr,
                             batch_size=batch)
    seconds = time.perf_counter() - t0

    curve = log["log"]
    best_corr = max(r["epoch"] for r in curve if r.get("best"))
    best_loss = min(curve, key=lambda r: r["valid_loss"])["epoch"]
    train_s = fitting.correlation(model, data, "train")
    valid_s = fitting.correlation(model, data, "valid")
    print(f"\nselected epoch {best_corr} by validation correlation {valid_s['mean_over_neurons']:.3f}; "
          f"the validation loss alone would have picked epoch {best_loss} "
          f"({[r for r in curve if r['epoch'] == best_loss][0]['valid_corr']:+.3f} correlation). "
          f"They disagree whenever the two criteria peak apart, which is why the spec names the "
          f"correlation.")

    RESULTS.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "widths": widths, "config": asdict(cfg),
                "epochs": epochs, "lr": lr, "batch_size": batch, "seed": seed,
                "best_epoch": best_corr, "valid_corr": valid_s["mean_over_neurons"]},
               RESULTS / "model.pt")
    (RESULTS / "train_log.json").write_text(json.dumps({
        "chosen": chosen, "n_parameters": model.n_parameters, "seconds": round(seconds, 1),
        "windows_train": len(data["train_set"]), "best_epoch_corr": best_corr,
        "best_epoch_loss": best_loss, "log": curve,
        "score_train": train_s, "score_valid": valid_s,
        "controls_valid": metrics.control_means(
            fitting.predict_split(model, data, "valid"),
            {k: data["y"][k] for k in data["split"]["valid"]}, bins=cfg.metric_bins),
    }, indent=2), encoding="utf-8")
    print(f"wrote results/model.pt ({sum(p.numel() for p in model.parameters()) / 1e6:.2f} M "
          f"parameters) and results/train_log.json")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
    figs.style()
    ep = [r["epoch"] for r in curve]
    axes[0].plot(ep, [r["train_loss"] for r in curve], color=figs.BLUE, lw=1.6, label="training")
    axes[0].plot(ep, [r["valid_loss"] for r in curve], color=figs.ORANGE, lw=1.6, label="validation")
    figs.annotate_peak(axes[0], best_loss, min(r["valid_loss"] for r in curve),
                       f"best loss, epoch {best_loss}", figs.MUTED)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("poisson NLL (normalized; can go negative)")
    axes[0].set_title("the criterion part1.train would use", loc="left", fontsize=9)
    axes[0].legend(fontsize=7)
    corr = [r["valid_corr"] for r in curve]
    axes[1].plot(ep, corr, color=figs.GREEN, lw=1.6, label="validation correlation")
    axes[1].axhline(0.455, color=figs.GREY, lw=1.0, ls=(0, (1, 2)))
    axes[1].annotate("smoothed-derivative baseline, 0.455", (ep[0], 0.455), fontsize=6.5,
                     color=figs.MUTED, va="bottom", xytext=(2, 2), textcoords="offset points")
    figs.annotate_peak(axes[1], best_corr, max(corr), f"kept, epoch {best_corr}")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation correlation")
    axes[1].set_title("the criterion this run uses", loc="left", fontsize=9)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    figs.save(fig, "training_curves.png")
    print(f"\n{seconds / 60:.1f} min of training on {part1.device}")


if __name__ == "__main__":
    main()
