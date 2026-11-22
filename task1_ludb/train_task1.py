"""Train the final Task 1 model:  python train_task1.py

Everything about the run - the cropping, the scaling, the class weights, the
widths of the U-Net, the learning rate, the number of epochs - comes from
results/choices.json, which validate_choices.py produced on the validation split.
The test split is not read by this script: it is not even loaded (the only place
it appears is the printed record count of the split dict).

Writes results/model.pt, results/train_log.json and figures/training_curves.png.
Re-running it with the same seed reproduces the same numbers.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import warnings

warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import ludb_data
import metrics
import part1
from ludb_data import Config, seed_everything
from part1 import UNet, predict, train

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"


def chosen_config() -> dict:
    path = RESULTS / "choices.json"
    if not path.exists():
        raise SystemExit("run `python validate_choices.py` first: it makes every choice on the "
                         "validation split and writes results/choices.json")
    return json.loads(path.read_text(encoding="utf-8"))["chosen"]


def train_model(chosen: dict | None = None, epochs: int | None = None) -> dict:
    """Train, save and report. Returns the log (also written to results/train_log.json)."""
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    chosen = chosen or chosen_config()
    epochs = epochs or chosen["epochs"]
    cfg = Config(**chosen["data"])
    seed_everything(chosen["seed"])

    data = ludb_data.build(cfg)
    print(ludb_data.summary(data))
    print(f"\nconfig from the validation split: {json.dumps(chosen)}")
    print(f"training on {len(data['split']['train'])} records, choosing the epoch on "
          f"{len(data['split']['valid'])} validation records; the test split stays untouched\n")

    weight = None if chosen["weights"] == "none" else torch.tensor(
        ludb_data.class_weights(data, power=float(chosen["weights"])), dtype=torch.float32)
    print(f"class weights: {'none' if weight is None else np.round(weight.numpy(), 3)} "
          f"(from the training records' frequencies only)\n")

    model = UNet(dim=1, c_in=1, c_out=4, widths=chosen["widths"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet(dim=1, c_in=1, c_out=4, widths={chosen['widths']}): "
          f"{n_params:,} parameters\n")
    sets = ludb_data.sets(data)
    seen_before = set(plt.get_fignums())
    t0 = time.perf_counter()
    model = train(model, sets["train"], sets["valid"],
                  nn.CrossEntropyLoss(ignore_index=-1, weight=weight),
                  epochs=epochs, lr=chosen["lr"], batch_size=chosen["batch_size"])
    seconds = time.perf_counter() - t0

    # the loss curves train() drew: saved as a figure and their numbers pulled back out,
    # so that part1.train() can stay exactly as the notebook gives it
    new = [plt.figure(n) for n in plt.get_fignums() if n not in seen_before]
    curves = {}
    if new:
        for line in new[-1].axes[0].get_lines():
            if line.get_label() in ("training", "validation"):
                curves[line.get_label()] = [round(float(v), 5) for v in line.get_ydata()]
        new[-1].set_size_inches(6, 3)
        new[-1].tight_layout()
        new[-1].savefig(FIGURES / "training_curves.png", dpi=150)
        plt.close("all")

    model.eval()
    torch.save({"state_dict": model.state_dict(), "config": chosen, "epochs": epochs},
               RESULTS / "model.pt")

    out, target = predict(model, sets["valid"])
    scores = metrics.evaluate(out.argmax(1).numpy(), target.numpy())
    baseline_scores = None
    print("\n" + metrics.table({"this model (validation)": scores}))
    log = {"chosen": chosen, "parameters": n_params, "epochs": epochs, "seconds": round(seconds, 1),
           "curves": curves, "validation_dice": scores,
           "validation_mean_dice": float(np.mean([scores[W]["global"] for W in ("P", "QRS", "T")])),
           "data_summary": ludb_data.summary(data)}
    (RESULTS / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"\n{epochs} epochs in {seconds:.0f} s; model -> results/model.pt, "
          f"log -> results/train_log.json, curves -> figures/training_curves.png")
    return log


if __name__ == "__main__":
    train_model()
