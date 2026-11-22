"""The specification Task 1 is built to, as text, so it is printed in the log and
not only promised in a notebook cell. `python spec.py` prints it; evaluate_task1.py
prints it before it reports any number.

The first table is the one filled in under Task 1's data cell in the notebook
(cell 48 of Copy_of_ex09_student.ipynb, unchanged here). The second table is what
the final run actually used - read from results/choices.json, which
validate_choices.py wrote from the validation split - so the printed specification
and the model that produced the numbers cannot drift apart.
"""
from __future__ import annotations

import json
from pathlib import Path

TABLE = """| | your answer |
| --- | --- |
| network input (shape) | shape (1, 5000) - lead II of one 10 s record, standardized per record. Under Option B the 5000 samples are the annotated span (median 7.3 s) left-aligned and padded back to 5000, so the shape in this row stays true |
| network output (shape) | shape (4, 5000) - for every one of the 5000 samples, one logit per class: background, P, QRS, T. `argmax` gives the class of every sample; the dense class map **is** the answer, not a list of wave positions |
| last layer | 1 x 1 convolution, 4 channels, **no activation** - the softmax is in the loss. This is `UNet`'s own head with `c_out=4`; no pooling, no `Linear` head |
| training loss | cross-entropy over the 4 classes, skipping the samples nobody annotated: `nn.CrossEntropyLoss(ignore_index=-1)` |
| evaluation metric | Dice of every wave (P, QRS, T) on the test set, counted over the annotated samples only, next to the baseline's; plus the QRS duration in ms, predicted against annotated |
| baseline | R peaks with `find_peaks` on the band-passed signal, then P, QRS and T placed at their median distances from the R peak, those distances measured on the **training** patients |

Assumptions behind the table: four classes per sample, as the task states them, so
4 output channels and cross-entropy. And the samples nobody annotated carry the
target `-1`: LUDB labels only about 7.3 s of each 10 s record, so the remaining
26.9% of the samples - 413 real heartbeats, about one in five - are not
background, they are unknown. Option B deals with them by cropping every record to
its annotated span, `ignore_index=-1` keeps the cropped-away part and the padding
out of the loss, and the metric counts only the annotated samples. Without that, a
perfect model cannot pass about 0.90 on QRS and the 0.89 baseline cannot be beaten."""


def text(width: int = 100) -> str:
    lines = ["TASK 1 SPECIFICATION", "=" * width, TABLE]
    path = Path(__file__).resolve().parent / "results" / "choices.json"
    if path.exists():
        chosen = json.loads(path.read_text(encoding="utf-8"))["chosen"]
        rows = [("data", f"Option B crop, margin {chosen['data']['margin']} samples, "
                         f"{chosen['data']['standardize']} statistics"),
                ("class weights", chosen["weights"]),
                ("network", f"UNet(dim=1, c_in=1, c_out=4, widths={chosen['widths']})"),
                ("loss", "CrossEntropyLoss(ignore_index=-1)"),
                ("optimizer", f"Adam, lr {chosen['lr']:g}, batch {chosen['batch_size']}"),
                ("epochs", f"{chosen['epochs']} trained, the best validation epoch kept"),
                ("seed", chosen["seed"]),
                ("baseline detector", json.dumps(chosen.get("baseline_params")
                                                 or chosen["baseline"]["params"]))]
        lines += ["", "AS BUILT - every hyperparameter chosen on the validation split",
                  "-" * width] + [f"  {name:<18} {value}" for name, value in rows]
    return "\n".join(lines)


if __name__ == "__main__":
    print(text())
