"""The specification Task 3 is built to, as text, so it is printed in the log and not
only promised in a notebook cell. `python spec.py` prints it; `evaluate_task3.py`
prints it before it reports any number.

The first table is the one filled in under Task 3's data cell in the notebook (cell 59
of Copy_of_ex09_student.ipynb, in condensed form - the notebook's copy carries the
measurements behind each row). The second table is what the final run actually used,
read from results/choices.json, which validate_choices.py wrote from the validation
split, so the printed specification and the model that produced the numbers cannot
drift apart.
"""
from __future__ import annotations

import json
from pathlib import Path

TABLE = """| | your answer |
| --- | --- |
| network input (shape) | shape (1, 512) - one neuron's fluorescence, resampled 100 Hz -> 25 Hz by averaging every 4 samples (86-100% of the power is below 5 Hz and <=0.7% above 25 Hz, so this throws nothing away), standardized per neuron from that neuron's own trace only. 512 samples = 20.5 s, 4x the context per layer 100 Hz would give, which the calcium kernel needs (0.4-2.3 s to fall back to half its peak). Windows are cut at the end of the real part, so none contains NaN padding |
| network output (shape) | shape (1, 512) - the predicted spike rate for the same window at the same 25 Hz: one rate per 40 ms bin, the metric's own resolution. Fully convolutional, every position predicted and used, nothing cropped |
| last layer | 1 x 1 convolution to 1 channel, then softplus - UNet's own head with c_out=1, no pooling and no Linear. A rate is non-negative and unbounded above; softplus keeps a gradient where 95.6% of the bins sit, where ReLU would be dead exactly at 0 |
| training loss | Poisson negative log-likelihood on the 40 ms counts, `rho - r log rho`, with the rate expressed in units of that neuron's own mean rate, averaged within a window and then over the batch. A plain MSE over concatenated windows is a different objective: 10 of the 174 neurons hold 30% of its weight |
| evaluation metric | the challenge's own, unchanged: sum predicted rate and true counts into 40 ms bins, Pearson correlation per test neuron, mean over the 34 test neurons. Reported next to the per-dataset means, the mean over the ten datasets instead of over neurons, the count of undefined (constant) predictions, and the +-0.005 noise floor of that mean |
| baseline | both classical guesses, rebuilt on this split rather than quoted, smoothing width tuned on the validation neurons only: the fluorescence itself and its smoothed positive derivative. The task text quotes 0.21 and 0.45 |

Assumptions behind the table: the given split and the 40 ms metric are kept exactly as
the data cell builds them. Three things the table cannot hold, because the audit found
them. Dataset 9 contains one neuron twice (columns 3 and 4 are the same recording,
spikes identical, one copy in train and one in test), so (9,3) is dropped from the
training set - 1 of 122 neurons, no test neuron changes. The model is selected on the
validation **correlation**, not on the loss: part1's train() early-stops on the loss,
and a Poisson loss and a Pearson correlation need not peak in the same epoch. And
nothing is tuned per dataset - five of the ten datasets have exactly one validation
neuron, so a per-dataset choice would be fitting one number."""


def text(width: int = 100) -> str:
    lines = ["TASK 3 SPECIFICATION", "=" * width, TABLE]
    path = Path(__file__).resolve().parent / "results" / "choices.json"
    if path.exists():
        chosen = json.loads(path.read_text(encoding="utf-8"))["chosen"]
        d = chosen["data"]
        rows = [("data", f"{d['decimate']}-sample mean to {100 // d['decimate']} Hz, "
                         f"{d['window']} samples = {d['window'] * d['decimate'] / 100:.1f} s, "
                         f"{d['standardize']} statistics, "
                         f"duplicate dropped: {d['drop_duplicate']}"),
                ("target", "counts per 40 ms bin, in units of the neuron's own mean rate"),
                ("network", f"UNet(dim=1, c_in=1, c_out=1, widths={chosen['widths']}) + softplus"),
                ("loss", "poisson_nll(rho - r*log rho), per-window then per-batch mean"),
                ("optimizer", f"Adam, lr {chosen['lr']:g}, batch {chosen['batch_size']}"),
                ("epochs", f"{chosen['epochs']} trained, the best validation-correlation epoch kept"),
                ("baseline", f"smoothed positive derivative, sigma {chosen['baseline_sigma']} "
                             f"samples ({chosen['baseline_sigma'] * 10} ms), tuned on validation"),
                ("seed", chosen["seed"]),
                ("selected on", "validation correlation over whole recordings; "
                                "the test split is read once, by evaluate_task3.py")]
        lines += ["", "AS BUILT - every hyperparameter chosen on the validation split",
                  "-" * width] + [f"  {name:<18} {value}" for name, value in rows]
    return "\n".join(lines)


if __name__ == "__main__":
    print(text())
