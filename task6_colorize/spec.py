"""The specification the code in this folder is built to, as printed text.

Cell 77 of `../Copy_of_ex09_student.ipynb` holds this table in prose, with the measurements
that produced each answer; this is its machine-readable restatement, and
`train_task6.py` / `evaluate_task6.py` print it before they print a single number, so the
specification on the page and the model that produced the figures cannot drift apart.

Below the table, `CONSTANTS` names the numbers the code enforces rather than restates -
`colorspace.CHROMA_SCALE`, `colorspace.GRID`, `metrics.CLIP_DB` - and `check_code_matches()`
compares the two. That is the drift worth checking: a constant changed in the code while the
table keeps its old text is exactly the mistake a reader cannot see.

    python spec.py
"""
from __future__ import annotations

import json
from pathlib import Path

import colorspace as cs
import metrics as M

HERE = Path(__file__).resolve().parent
CHOICES = HERE / "results" / "choices.json"

TABLE = """| | A - regression | B / B+ - bins |
| --- | --- | --- |
| network input (shape) | `(B, 1, 64, 64)` float32 in [0, 1]: the luminance `Y = rgb @ [0.299, 0.587, 0.114] / 255` of the 64x64 centre crop - one channel, and the *same* Y that defines the target | same |
| network output (shape) | `(B, 2, 64, 64)`: (Cb, Cr) with neutral at 0, in units of +-128 (divide by 128, so the target sits inside +-0.13 and a zero-initialised head starts exactly at the gray baseline) | `(B, 256, 64, 64)` logits: a 16x16 grid on (Cb, Cr). Not 36 bins - a 36-colour vocabulary caps the PSNR at 22.3 dB with a perfect classifier |
| last layer | 1x1 conv, **no activation**. RGB is rebuilt as `rgb = ycc_inv(Y, 128 * out)` with **Y written back unchanged**, then clamped to [0, 255] | 1x1 conv to 256 channels; the softmax lives inside the loss. **B** = `argmax p`; **B+** = `argmax(log p - gamma * log pbar)` at inference only |
| training loss | **L1** on the two chroma planes, plus **one scalar alpha calibrated on valid** (the uncalibrated optimum keeps 29% of the colour, which *is* the task's "dull brownish gray") | cross-entropy on the 256 bin labels, unweighted. The prior pull belongs at inference only |
| evaluation metric | PSNR (per photo, each photo clipped at 40 dB, with median and pooled beside it), chroma L1, colourfulness - from one function, `metrics.score` - plus photos beating gray and the split by colour quintile | the same three, plus top-1 bin accuracy **only ever against its ceiling at this grid - 30.1%, measured on the training photos by `metrics.ceilings`; the audit's better-known 18.8% is the same lookup table at a 32x32 chroma grid** - and that ceiling bounds a colour *lookup table* (chroma read off one 64-bin number per pixel), not a network, so beating it is a result rather than a contradiction. Never a headline on its own |
| baseline | the gray input (chroma = 0), conventions attached: test 25.36 dB mean, 23.21 median, 24.06 clipped at 40, 21.11 pooled, chroma L1 0.0804, colourfulness 0.0000 | the same baseline, plus the ceiling of the binned truth at 256 bins (31.54 dB on the 100 valid photos) - the one hard ceiling here: no decoder that can only emit bin centres passes it |"""

CONSTANTS = {
    "chroma units (neutral 0, +-128)": ("colorspace.CHROMA_SCALE", cs.CHROMA_SCALE),
    "bin grid (route B)": ("colorspace.GRID", cs.GRID),
    "bins = grid squared": ("grid x grid", cs.GRID * cs.GRID),
    "PSNR clip per photo, dB": ("metrics.CLIP_DB", M.CLIP_DB),
    "MSE floor (sets the 120 dB ceiling)": ("metrics.EPS", M.EPS),
    "luminance weights": ("colorspace.Y_W", [float(w) for w in cs.Y_W]),
    "bootstrap draws / seed": ("metrics.DRAWS / SEED", (M.DRAWS, M.SEED)),
}


def _json(path: Path) -> dict:
    """Read a results file if it exists; the specification prints before any of them do."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def check_code_matches() -> list[str]:
    """The four numbers the table states and the code enforces, compared rather than trusted."""
    lines = []
    for what, (where, value) in CONSTANTS.items():
        lines.append(f"  {what:<34} {where:<26} {value}")
    expected = {"bin grid (route B)": 16, "bins = grid squared": 256,
                "chroma units (neutral 0, +-128)": 128.0, "PSNR clip per photo, dB": 40.0}
    for what, want in expected.items():
        got = CONSTANTS[what][1]
        assert got == want, (f"{what}: the code says {got}, the specification table says {want} - "
                             f"one of them has drifted")
    return lines


def text(width: int = 100) -> str:
    lines = ["TASK 6 SPECIFICATION - colourisation from luminance", "=" * width, TABLE,
             "", "Assumptions behind the table:",
             "  * the input is the photo's own luminance and the output is added to it; the network",
             "    never predicts Y, so 99% of the signal cannot be damaged by it (spec item 2);",
             "  * colourfulness means mean |chroma| over pixels and channels, /255 - the definition",
             f"    whose value on the real photos is the task's 0.04 (it gives {0.0407:.4f});",
             "  * no colour augmentation exists in this pipeline: jitter, channel shuffle and",
             "    autocontrast would edit the target (spec item 8);",
             "  * `valid` (100 photos, the handout's split) decides every choice; `valrest` (the 3325",
             "    photos the handout leaves unused) is a confirmation column only; `test` (500) is",
             "    read once, by evaluate_task6.py, and decides nothing.",
             "", "The constants the code enforces rather than restates:",
             *check_code_matches()]
    if not CHOICES.exists():
        return "\n".join(lines)
    chosen = json.loads(CHOICES.read_text(encoding="utf-8"))["chosen"]
    cal = _json(HERE / "results" / "calibration.json")
    tlog = _json(HERE / "results" / "train_log.json")
    lines += ["", "AS BUILT - every hyperparameter chosen on the validation split", "-" * width]

    def epoch_of(route: str) -> str:
        if route in tlog:
            return (f"{tlog[route]['best_epoch']} kept of {tlog[route]['epochs']} trained "
                    f"(best validation loss {tlog[route]['best_valid_loss']}, "
                    f"{tlog[route]['seconds'] / 60:.1f} min)")
        return f"budget {chosen[route]['epochs_budget']}, not trained yet"

    rows = [("data", "imagenette2-160, the notebook's data cell: short side 64 with BICUBIC, "
                     "central 64x64, seed-0 permutation of val/"),
            ("input / target", "Y/255 -> (Cb, Cr)/128 for A; the same Y -> the 16x16 bin "
                               "index for B"),
            ("network A", f"UNet(dim=2, c_in=1, c_out=2, widths={chosen['A']['widths']}), "
                          "zeroed 1x1 head"),
            ("network B", f"UNet(dim=2, c_in=1, c_out={cs.GRID ** 2}, "
                          f"widths={chosen['B']['widths']}), head bias = log(train bin marginal)"),
            ("loss", "A: L1 on both chroma planes;  B: cross-entropy on the bin labels over "
                     f"{'all' if chosen['B'].get('pixels') is None else chosen['B']['pixels']} "
                     "sampled pixels per photo per step"),
            ("optimizer", "Adam, as in the notebook's `train`"),
            ("epochs", f"A: {epoch_of('A')};  B: {epoch_of('B')}"),
            ("flip arm", f"A: {chosen['A']['flip']}, B: {chosen['B']['flip']} "
                         "(measured as an arm, not assumed)"),
            ("alpha (route A)", f"{cal.get('alpha', {}).get('chosen', 'not calibrated yet')}"
                                f" - {cal.get('alpha', {}).get('rule', 'rule CA in calibrate.py')}"),
            ("gamma (route B+)", f"{cal.get('gamma', {}).get('chosen', 'not calibrated yet')}"
                                 " - the same rule CA, on valid"),
            ("seed", chosen.get("seed", 0)),
            ("selected on", "validation loss for the epoch; the 100 validation photos for "
                            "alpha, gamma, width, lr, batch, pixels; the test split for nothing")]
    lines += [f"  {name:<18} {value}" for name, value in rows]
    return "\n".join(lines)


if __name__ == "__main__":
    print(text())
