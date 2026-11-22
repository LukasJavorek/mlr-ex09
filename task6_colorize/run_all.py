"""Run all of Task 6 in order. `python run_all.py`: 89.1 min measured on 6 CPU threads -
sweep 49.0, training 36.2, calibration 0.8, the two evaluations 2.7, the rest 0.4.

    python run_all.py                      everything
    python run_all.py --quick              half the sweep's photos and half its epochs (2/3/2
                                       instead of 4/6/4), nothing else. Expect ~53 min, which is
                                       arithmetic: the sweep's own docstring puts its quick mode
                                       at ~13 min, the other six steps here are measured at 40.1
                                       min, and this combination has not been timed as a whole.
                                       The training does not shrink, which is where most of the
                                       remaining time actually is
    python run_all.py --skip-choices       reuse results/choices.json (the sweep is the slow part)
    python run_all.py --until 4            stop after step 4 (calibrate), leaving the test
                                           split untouched

Each step is a separate process, so torch's memory is released between them, and each one's
console output is written three times - to the screen, to `results/<name>_log.txt`, and into the
one combined `results/run_all_log.txt` - by the loop below, at full budget. Those seven files are
therefore the output of exactly the command printed above each of them, and nothing else; the
step's own header line goes to the combined stream only, so no log contains a line the step did
not print. The combined file is written by this script rather than by a shell pipe, so the two
cannot come from different runs: the last lines below assert that all seven appear in the
combined file, in order, byte for byte, and say so.

Step 7 is the only step that scores the test photos, and `--until 6` or less stops before it.
(An earlier draft of this line said step 6 and `--until 5`: step 6 scores the *valid* photos, so the
real boundary is one step later, and the guard at the bottom already keyed on 7.)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# (step, what it decides, command, gate, name of the log the step's output is written to)
STEPS = [
    (1, "the arrays and their inventory", ["col6_data.py"], None, "data"),
    (2, "the smoke test: every check, nothing trained", ["smoke_test.py"], None, "smoke"),
    (3, "the choices sweep on the validation split", ["validate_choices.py"], "choices", "sweep"),
    (4, "train both routes on all 9469 training photos", ["train_task6.py"], None, "train"),
    (5, "calibrate alpha and gamma on valid", ["calibrate.py"], None, "calibrate"),
    (6, "evaluate on valid (dry run)", ["evaluate_task6.py", "--valid", "--no-valrest"], None,
     "eval_valid"),
    (7, "evaluate on the test split - the one look", ["evaluate_task6.py"], None, "eval_test"),
]


def tee(cmd: list[str], env: dict, log: str, combined) -> int:
    """Run one step, copying its console output to the screen, to `log` and to `combined`.

    Written as a tee rather than as `subprocess.run(...)` so that the shipped log cannot be a
    reconstruction: it is the bytes the step printed, in the order it printed them, and a step
    that dies part-way leaves its own log ending at the line it died on. Writing the combined
    file here as well, instead of piping this script into `tee`, is what makes the seven and the
    one come from the same run of the same code - `verify()` below then checks it.
    """
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fh.write(line)
            combined.write(line)
            combined.flush()
        return p.wait()


def verify(names: list[str], combined_path: str) -> bool:
    """Every step's own log must sit inside the combined log, in order, byte for byte."""
    blob = open(combined_path, encoding="utf-8").read()
    at, bad, own = 0, [], 0
    for n in names:
        text = open(os.path.join(HERE, "results", f"{n}_log.txt"), encoding="utf-8").read()
        i = blob.find(text, at)
        if i < 0:
            bad.append(n)
        else:
            at = i + len(text)
            own += len(text)
    if bad:
        print(f"LOG ASSEMBLY FAILED for: {', '.join(bad)} - the per-step file is not a verbatim "
              f"block of {os.path.basename(combined_path)}, so one of the two is stale")
        return False
    print(f"log assembly verified: the {len(names)} per-step logs appear in "
          f"{os.path.basename(combined_path)} in order, byte for byte - {own} of its "
          f"{len(blob)} characters, the rest being this script's own headers and summary")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-choices", action="store_true")
    ap.add_argument("--until", type=int, default=len(STEPS), choices=range(1, len(STEPS) + 1))
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONUTF8": "1"}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    combined_path = os.path.join(HERE, "results", "run_all_log.txt")
    combined = open(combined_path, "w", encoding="utf-8")

    def say(text: str = "") -> None:
        """A line this script prints itself - a header or a summary, never a step's output."""
        print(text, flush=True)
        combined.write(text + "\n")
        combined.flush()

    t_all = time.perf_counter()
    timings, ran = [], []
    say(f"Task 6, end to end. {len(STEPS)} steps, stopping after {a.until}"
        + ("  [--quick]" if a.quick else "")
        + ("  [--skip-choices]" if a.skip_choices else ""))
    for n, what, args, gate, logname in STEPS:
        if n > a.until:
            say(f"\n[{n}/{len(STEPS)}] {what}: not run (--until {a.until})")
            continue
        if gate == "choices" and a.skip_choices:
            say(f"\n[{n}/{len(STEPS)}] {what}: skipped (--skip-choices)")
            continue
        cmd = [PY, "-X", "utf8"] + args + (["--quick"] if a.quick and "validate" in args[0] else [])
        say(f"\n{'#' * 100}\n# [{n}/{len(STEPS)}] {what}\n#   {' '.join(cmd[2:])}\n{'#' * 100}")
        t0 = time.perf_counter()
        rc = tee(cmd, env, os.path.join(HERE, "results", f"{logname}_log.txt"), combined)
        mins = (time.perf_counter() - t0) / 60
        timings.append((n, args[0], round(mins, 1)))
        ran.append(logname)
        say(f"# [{n}/{len(STEPS)}] {'done' if rc == 0 else f'FAILED ({rc})'} in {mins:.1f} min"
            f"  ->  results/{logname}_log.txt")
        if rc != 0:
            say("stopping: a step failed, so nothing after it can be trusted")
            break
    say(f"\n{'step':>6} {'script':<26} {'min':>7}")
    for n, script, mins in timings:
        say(f"{n:>6} {script:<26} {mins:>7.1f}")
    say(f"{'':>6} {'total':<26} {sum(t for _, _, t in timings):>7.1f}")
    say(f"wall clock {(time.perf_counter() - t_all) / 60:.1f} min")
    if a.until < 7:
        say("the test photos were never scored in this run")
    combined.close()
    if not verify(ran, combined_path):
        sys.exit(3)


if __name__ == "__main__":
    main()
