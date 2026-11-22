"""Task 1 from an empty folder to the final numbers:

    python run_all.py                 # ~2 h on a 6-thread CPU; step 2 alone is 1 h 45 m
    python run_all.py --quick         # smaller sweeps, ~50 min
    python run_all.py --skip-choices  # reuse results/choices.json, ~20 min

In order:
  0  download and read LUDB (15 MB, once; the arrays are cached in data/)
  1  the smoke test - shapes, loss, class mapping, masking, capacity
  2  every choice, made on the validation split        -> results/choices.json
     (stages A-D, then E, F and G: epochs, batch size, depth, and the depth again
      measured at the length the final run actually trains for)
  3  train the model with those choices                -> results/model.pt
  4  one evaluation on the test split                  -> figures/ and results/

The test split is read in step 4 only. Each step is also a script of its own
(ludb_data.py, smoke_test.py, validate_choices.py, train_task1.py,
evaluate_task1.py) if you want to run or re-run one of them alone.
"""
from __future__ import annotations

import sys
import time
import warnings

warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

import matplotlib
matplotlib.use("Agg")

import evaluate_task1
import ludb_data
import smoke_test
import train_task1
import validate_choices


def step(n: str):
    print(f"\n{'#' * 78}\n# step {n}\n{'#' * 78}", flush=True)
    return time.perf_counter()


def main() -> None:
    quick = "--quick" in sys.argv
    skip_choices = "--skip-choices" in sys.argv
    timings = {}

    t = step("0  the data (Option B: every record cropped to its annotated span)")
    data = ludb_data.build(ludb_data.configured())   # what choices.json picked, or the defaults
    print(ludb_data.summary(data))
    for name, idx in data["split"].items():
        print(f"  {name:<6} {len(idx)} records: "
              f"{[int(r) for r in data['record'][idx]][:8]}{' ...' if len(idx) > 8 else ''}")
    timings["data"] = time.perf_counter() - t

    t = step("1  smoke test")
    smoke_test.main()
    timings["smoke"] = time.perf_counter() - t

    t = step("2  the choices, made on the validation split (the test split is not read)")
    if skip_choices:
        print("skipped on request: reusing results/choices.json")
    else:
        validate_choices.main(quick=quick)
        validate_choices.refine(quick=quick)        # stage E and F: epochs, batch, depth
        validate_choices.confirm_depth(quick=quick)  # stage G: depth, at the real length
    timings["choices"] = time.perf_counter() - t

    t = step("3  training on the 140 training records")
    train_task1.train_model()
    timings["train"] = time.perf_counter() - t

    t = step("4  the single evaluation on the 30 test records")
    evaluate_task1.main()
    timings["test"] = time.perf_counter() - t

    print(f"\n{'#' * 78}\ndone in {sum(timings.values()) / 60:.1f} min: "
          + ", ".join(f"{k} {v / 60:.1f}" for k, v in timings.items()))
    print("figures/  - validation_choices.png, training_curves.png, test_dice.png, example_record.png")
    print("results/  - choices.json, train_log.json, test_scores.json, model.pt")


if __name__ == "__main__":
    main()
