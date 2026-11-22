"""Train the two models on all 9469 training photos. `python train_task6.py`, 35.9 min
measured: route A 16 min at 60 s/epoch, route B 20 min at 74 s/epoch, both on 6 CPU threads.

Reads the configuration `validate_choices.py` measured (widths, learning rate, batch size,
the flip arm, route B's pixel count) and re-trains each route once on the full training split
for a budget of `epochs_budget` epochs, keeping the epoch with the best validation loss.

Why the epoch count is not swept separately: the best epoch depends on how much data each
epoch sees, so an epoch sweep at a quarter of the data would pick the wrong epoch for the run
that matters. Training a budget and keeping its best-loss epoch is the same measurement, done
on the model that gets used, and it costs nothing extra - the per-epoch validation loss is
logged anyway.

Both routes are scored on the validation photos after training, at alpha = 1 and gamma = 0,
because "how did it do" on valid is a training output. The calibration of alpha and gamma is
NOT done here - those are properties of a trained model's outputs, so they come after this
script, in `calibrate.py`. The test split is never read: this file builds train and valid
datasets only, and `d6.build` is called with the splits this script needs.

    python train_task6.py                     both routes
    python train_task6.py --only A            one route
    python train_task6.py --epochs 8          override the budget (for a fast look)
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

import colorspace as cs
import col6_data as d6
import figs
import fitting
import metrics as M
import spec

OUT = d6.HERE / "results"


def score_valid(model, valid_rgb: np.ndarray, prior) -> dict:
    """The trained model on the 100 validation photos, uncalibrated."""
    out = fitting.decode(model, valid_rgb, prior, alpha=1.0, gamma=0.0)
    return M.score(out["rgb"], valid_rgb, baseline_per_photo=M.baseline_per_photo(valid_rgb))


def train_route(route: str, data: dict, chosen: dict, epochs: int, prior) -> dict:
    train_rgb, valid_rgb = d6.arrays(data, "train"), d6.arrays(data, "valid")
    cfg = d6.Config(**{**data["cfg"], "flip": bool(chosen[route]["flip"])})
    tr = d6.Photos(train_rgb, route, cfg)
    va = d6.Photos(valid_rgb, route, d6.Config(**data["cfg"]))
    model = fitting.Colorizer(route, chosen[route]["widths"],
                              prior_prob=prior if route != "A" else None)
    print(f"\nroute {route}: UNet(dim=2, c_in=1, c_out={'2' if route == 'A' else cs.GRID ** 2}, "
          f"widths={chosen[route]['widths']}), {model.n_parameters()} parameters")
    px = chosen[route].get("pixels")
    print(f"  {len(tr)} training photos, flip={cfg.flip}, lr={chosen[route]['lr']:g}, "
          f"batch={chosen[route]['batch_size']}, "
          + ("loss = L1 on (Cb, Cr)/128" if route == "A" else
             f"loss = cross-entropy on {cs.GRID ** 2} bins over "
             f"{'all' if px is None else px} sampled pixels per photo per step, "
             f"head bias = log(train marginal)"))
    model, info = fitting.fit(model, tr, va, prior_prob=prior if route != "A" else None,
                              epochs=epochs, lr=chosen[route]["lr"],
                              batch_size=chosen[route]["batch_size"],
                              pixels=None if route == "A" else chosen[route]["pixels"],
                              seed=chosen[route]["seed"], verbose=True)
    fitting.save_model(model, OUT / f"model_{route}.pt",
                       prior_prob=prior if route != "A" else None,
                       lr=chosen[route]["lr"], batch_size=chosen[route]["batch_size"],
                       pixels=chosen[route]["pixels"], flip=cfg.flip,
                       n_train=len(tr), best_valid_loss=info["best_valid_loss"],
                       best_epoch=info["best_epoch"])
    info["valid_uncalibrated"] = score_valid(model, valid_rgb, prior)
    info["config"] = dict(chosen[route])
    curve_figure(route, info)
    return info


def curve_figure(route: str, info: dict) -> None:
    """Validation loss beside validation PSNR and colourfulness, for one trained route.

    The point of the figure is the disagreement: the epoch the loss picks and the epoch the
    task's headline metric peaks at are not the same epoch, and on this data the PSNR curve
    peaks where the colour is smallest.
    """
    curve = info["curve"]
    eps = np.arange(1, len(curve["valid_loss"]) + 1)
    figs.style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.7))
    axes[0].plot(eps, curve["valid_loss"], marker="o", ms=3, color=figs.METHOD_COLORS["A"],
                 label="validation loss")
    figs.annotate_peak(axes[0], info["best_epoch"], max(curve["valid_loss"]),
                       f"epoch {info['best_epoch']} kept")
    axes[0].set_title("validation loss (what selects the epoch)", fontsize=8, loc="left")
    axes[0].set_xlabel("epoch", fontsize=7)
    decode = "alpha = 1" if route == "A" else "argmax p"
    axes[1].plot(eps, curve["valid_psnr"], marker="o", ms=3, color=figs.METHOD_COLORS["B"],
                 label="validation PSNR")
    peak = int(np.argmax(curve["valid_psnr"])) + 1
    figs.annotate_peak(axes[1], peak, max(curve["valid_psnr"]), f"PSNR peaks at epoch {peak}",
                       color=figs.BLUE)
    axes[1].set_title(f"validation PSNR ({decode})", fontsize=8, loc="left")
    axes[1].set_xlabel("epoch", fontsize=7)
    axes[2].plot(eps, curve["valid_colourfulness"], marker="o", ms=3,
                 color=figs.METHOD_COLORS["B+"], label="colourfulness")
    axes[2].axhline(info["valid_uncalibrated"]["colourfulness_truth"], color=figs.GREY,
                    linestyle="--", linewidth=1.0,
                    label=f"the photos' own {info['valid_uncalibrated']['colourfulness_truth']:.4f}")
    axes[2].axhline(0, color=figs.GREY, linewidth=0.8, label="gray (no colour)")
    axes[2].set_title(f"validation colourfulness ({decode})", fontsize=8, loc="left")
    axes[2].set_xlabel("epoch", fontsize=7)
    for ax in axes:
        ax.tick_params(labelsize=7)
        figs.legend_on(ax, loc="lower right")
    fig.suptitle(f"route {route}: the epoch the loss keeps and the epoch PSNR peaks at are "
                 f"different - which is the task's trap", fontsize=8.5)
    figs.save(fig, f"training_curve_{route}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=("A", "B"), help="train one route")
    ap.add_argument("--epochs", type=int, help="override the epoch budget")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    fitting.threads(6)
    d6.seed_everything(0)
    t_start = time.perf_counter()

    print(spec.text())
    choices = json.loads((OUT / "choices.json").read_text(encoding="utf-8"))
    chosen = choices["chosen"]
    print(f"\nusing results/choices.json: {json.dumps({r: {k: chosen[r][k] for k in
          ('widths', 'lr', 'batch_size', 'flip', 'pixels')} for r in ('A', 'B')}, indent=None)}")

    data = d6.build(verbose=False)
    print("\nthe split this script is allowed to see: train and valid. "
          f"({len(data['split']['test'])} test photos exist and are not loaded into any "
          f"dataset here.)")
    prior = fitting.prior(d6.arrays(data, "train"), cs.GRID)
    print(f"route B prior: {-(prior * np.log(prior)).sum():.4f} nats entropy, modal cell "
          f"{prior.max() * 100:.2f}% of the training pixels, "
          f"{int((prior > 0).sum())} of {cs.GRID ** 2} cells populated")

    gray_mean = float(M.psnr_photos(M.baseline_photos(d6.arrays(data, "valid")),
                                    d6.arrays(data, "valid")).mean())
    log = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "threads": 6,
           "gray_valid_psnr": round(gray_mean, 3),
           "torch": torch.__version__, "numpy": np.__version__}
    for route in (("A", "B") if a.only is None else (a.only,)):
        epochs = a.epochs or chosen[route]["epochs_budget"]
        log[route] = train_route(route, data, chosen, epochs, prior)
        v = log[route]["valid_uncalibrated"]
        print(f"  route {route} on valid, uncalibrated: {v['psnr_mean']:.2f} dB mean against "
              f"gray's {gray_mean:.2f}, colourfulness {v['colourfulness']:.4f} against the "
              f"photos' own {v['colourfulness_truth']:.4f}, beating gray on "
              f"{v['beats_baseline_pct']}% of the 100")

    (OUT / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"\nwrote results/train_log.json and results/model_*.pt in "
          f"{(time.perf_counter() - t_start) / 60:.1f} min")
    print("next: python calibrate.py (alpha and gamma, on valid), then "
          "python evaluate_task6.py (the one look at test)")


if __name__ == "__main__":
    main()
