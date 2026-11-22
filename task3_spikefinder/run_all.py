"""Task 3 from an empty folder to the final numbers:

    python run_all.py                 # ~1 h 30 m on a 6-thread CPU; the sweeps are 87 min
    python run_all.py --quick         # 1 seed per arm, shorter arms, no stage 3: ~15 min
    python run_all.py --skip-choices  # reuse results/choices.json: ~2 min

Those totals are the measured stages added up - 17.4 + 44.2 + 25.2 min of sweep, printed
by the sweep itself; 88 s of training, logged; 6.5 s of smoke test and 6 s of evaluation,
wall-clock - not one stopwatch on this script. `--quick`'s total is the same arithmetic
after shortening the arms: it has never been run end to end, because running it would
overwrite the results/choices.json the shipped numbers come from.

In order:
  0  download and read spikefinder (71 MB zip, once; the decimated arrays are cached
     in data/) and print what the 174 neurons look like on the loader's grid
  1  the smoke test - nine checks: the grid, the split, no padding, the shapes, the
     loss identity, the metric's invariances, capacity, determinism, the softplus head
  2  every choice, made on the validation split        -> results/choices.json
     stage 1 the data (grid, window, standardization, target scale), stage 2 the
     optimizer (lr, depth, batch) and then the epoch count off a long curve, stage 3
     the winner at full length beside the runner-up and beside the reopened leak.
     What wins is not the argmax: `vc.pick()` puts the ablation arms out of contention,
     gives lr and batch to the cheaper arm inside the +-0.005 noise floor of an
     18-neuron mean, and gives depth to the measured receptive field (kernel half-decay
     <= RF <= window) rather than to +0.004 of seed noise. `vc.rechoose()` re-applies
     that rule to the recorded arms before anything is written.
  3  train the model with those choices                -> results/model.pt
  4  one evaluation on the test split                  -> figures/ and results/

The test split is read in step 4 only - by `evaluate_task3.py`, the one file in this
folder that names `split["test"]`. Each step is also a script of its own (sf_data.py,
smoke_test.py, validate_choices.py, train_task3.py, evaluate_task3.py) if you want to
run or re-run one alone; `baseline.py` and `spec.py` run on their own too.

`--quick` is for a first look, not for the shipped numbers: one seed per arm cannot
separate settings whose true difference is under the +-0.045 noise of 18 validation
neurons, and stage 3 - the only stage that runs at the length the final model actually
trains - is dropped.
"""
from __future__ import annotations

import sys
import time
import warnings

warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

import matplotlib
matplotlib.use("Agg")

import evaluate_task3
import sf_data
import smoke_test
import train_task3
import validate_choices as vc

QUICK = {"epochs": 8, "seeds": 1, "long": 40}
FULL = {"epochs": vc.SHORT, "seeds": 3, "long": vc.LONG}


def step(n: str):
    print(f"\n{'#' * 78}\n# step {n}\n{'#' * 78}", flush=True)
    return time.perf_counter()


def main() -> None:
    quick = "--quick" in sys.argv
    skip_choices = "--skip-choices" in sys.argv
    timings = {}

    t = step("0  the data: 25 Hz, per-neuron z-score, no padding, the duplicate dropped")
    data = sf_data.build(sf_data.configured())       # what choices.json picked, or the defaults
    print(sf_data.summary(data))
    for name, idx in data["split"].items():
        print(f"  {name:<6} {len(idx)} neurons: {sorted(idx)[:6]}{' ...' if len(idx) > 6 else ''}")
    timings["data"] = time.perf_counter() - t

    t = step("1  smoke test")
    smoke_test.main()
    timings["smoke"] = time.perf_counter() - t

    t = step("2  the choices, made on the validation split (the test split is not read)")
    if skip_choices:
        print("skipped on request: reusing results/choices.json")
    else:
        kw = QUICK if quick else FULL
        state = vc.start()
        seeds = range(kw["seeds"])
        vc.stage_one(state, kw["epochs"], seeds)
        vc.stage_two(state, kw["epochs"], seeds, long=kw["long"])
        vc.rechoose(state)            # the tie-break rule, applied to every sweep alike
        vc.save(state)
        if quick:
            print("\nstage 3 (the full-length confirmation) skipped by --quick; the shipped "
                  "numbers come from the unshortened `python run_all.py`")
        else:
            vc.stage_three(state, seeds)
    timings["choices"] = time.perf_counter() - t

    t = step("3  training on the 121 training neurons")
    train_task3.main()
    timings["train"] = time.perf_counter() - t

    t = step("4  the single evaluation on the 34 test neurons")
    evaluate_task3.main()
    timings["test"] = time.perf_counter() - t

    print(f"\n{'#' * 78}\ndone in {sum(timings.values()) / 60:.1f} min: "
          + ", ".join(f"{k} {v / 60:.1f}" for k, v in timings.items()))
    print("figures/  - validation_choices.png, training_curves.png, test_scores.png, "
          "example_neuron.png")
    print("results/  - choices.json, train_log.json, test_scores.json, model.pt")


if __name__ == "__main__":
    main()
