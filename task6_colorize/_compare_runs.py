"""Compare the shipped run against the previous one, leaf by leaf.

    python _compare_runs.py               results/ against _prev_run/results/
    python _compare_runs.py a b           any two results directories

The leading underscore is load-bearing, not shyness: `smoke_test.py`'s binding check scans
this folder's `*.py` files and skips the ones that start with `_`, because this file is a
tool *about* the run rather than a step *in* it. That is what lets `results/smoke_log.txt`
go on saying "13 scripts" after this file was added to the folder - the count means
"pipeline scripts", and it is a number a log printed, so it must not drift with the
folder's contents.

The second end-to-end run exists to answer one question: are the shipped numbers the
product of the code, or of the particular minute the machine happened to be in? Every
quantity the pipeline reports is a deterministic function of the arrays and the seeds, so
the answer should be "the code" - with the exception of the fields that *are* wall clock,
listed in TIMING below and reported separately rather than skipped silently.

Nothing here retrains anything; it only reads two sets of JSON, and it prints no number of
its own beyond a count of differences. Prose in the logs - a docstring line edited between
the two runs - shows up as a difference it cannot know is harmless, so each log's non-clock
lines are printed in full rather than summarised away.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.join(HERE, "results")
OLD = os.path.join(HERE, "_prev_run", "results")

TIMING = {"seconds", "started", "secs", "elapsed", "quick"}
# Any field whose name ends in _seconds is a duration too - `epoch_seconds` is a list of
# per-epoch wall clock, and calling those numeric differences would make the verdict say
# "DISAGREE" about a machine that was busy. The values are still printed, under [time].
DURATION_SUFFIX = "_seconds"


def is_timing(path: str) -> bool:
    leaf = path.rsplit(".", 1)[-1].split("[")[0]
    return leaf in TIMING or leaf.endswith(DURATION_SUFFIX)


def walk(a, b, path=""):
    """Yield (path, old, new) for every leaf whose value differs."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                yield (f"{path}.{k}", "<absent>", "<present>")
            elif k not in b:
                yield (f"{path}.{k}", "<present>", "<absent>")
            else:
                yield from walk(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield (f"{path}.len", len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            yield from walk(x, y, f"{path}[{i}]")
    elif a != b:
        yield (path, a, b)


def load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main() -> int:
    global NEW, OLD
    if len(sys.argv) == 3:
        NEW, OLD = sys.argv[1], sys.argv[2]
    for d in (NEW, OLD):
        if not os.path.isdir(d):
            print(f"no such directory: {d}  - this compares two results/ directories and "
                  f"retrains nothing; the default pair is results/ and _prev_run/results/")
            return 1
    names = sorted(f for f in os.listdir(OLD) if f.endswith(".json"))
    names += [f for f in sorted(os.listdir(NEW)) if f.endswith(".json") and f not in names]
    worst = bad_logs = 0
    for name in names:
        o, n = load(OLD, name), load(NEW, name)
        if o is None or n is None:
            print(f"{name:<28} only in {'new' if o is None else 'previous'} run")
            worst = max(worst, 1)
            continue
        diffs = list(walk(o, n))
        timing = [(p, x, y) for p, x, y in diffs if is_timing(p)]
        real = [(p, x, y) for p, x, y in diffs if (p, x, y) not in timing]
        print(f"{name:<28} {'DIFFERS' if real else 'IDENTICAL'}"
              f"   ({len(real)} numeric, {len(timing)} wall-clock)")
        for p, x, y in timing[:6]:
            print(f"      [time] {p}: {x} -> {y}")
        if len(timing) > 6:
            print(f"      [time] ... {len(timing) - 6} more duration fields, all of the same kind")
        for p, x, y in real[:40]:
            print(f"      [DIFF] {p}: {x!r} -> {y!r}")
        worst = max(worst, 1 if real else 0)
    # The logs print wall clock inline as well as numbers, so a line that differs only in a
    # "57.3 s" or a "(16 s" is the same line. Anything else that differs is a real difference.
    import difflib
    import re
    clock = re.compile(r"\d+(?:\.\d+)?\s*(?:s|sec|min|hours)\b|\(\d+ s|\d+/\d+ s")
    logs = [f for f in sorted(os.listdir(OLD))
            if f.endswith("_log.txt") and not f.startswith("dryrun")
            and f not in ("run_all_log.txt", "choices_report_log.txt")]
    print()
    for f in logs:
        a, b = os.path.join(OLD, f), os.path.join(NEW, f)
        if not (os.path.exists(a) and os.path.exists(b)):
            print(f"{f:<28} missing on one side")
            bad_logs += 1
            continue
        ta = open(a, encoding="utf-8").read().splitlines()
        tb = open(b, encoding="utf-8").read().splitlines()
        if ta == tb:
            print(f"{f:<28} IDENTICAL  ({len(ta)} lines, byte for byte)")
            continue
        diff = [l for l in difflib.unified_diff(ta, tb, lineterm="", n=0)
                if l[:1] in "+-" and not l.startswith(("+++", "---"))]
        # `smoke_test.py` draws its three figures into a fresh mkdtemp(), so the directory's
        # random name is in four lines of its log. Same files, same sizes, different name.
        temp = re.compile(r"task6_figcheck_\w+")
        timed = [l for l in diff if clock.search(l[1:])]
        named = [l for l in diff if l not in timed and temp.search(l[1:])]
        real = [l for l in diff if l not in timed and l not in named]
        print(f"{f:<28} {'IDENTICAL' if not real else 'DIFFERS'}  "
              f"({len(ta)} vs {len(tb)} lines, {len(diff)} differ: {len(timed)} wall clock, "
              f"{len(named)} temp-dir name, {len(real)} neither)")
        for l in named[:2]:
            print("      [temp]", l[:180])
        for l in real[:12]:
            print("      [DIFF]", l[:190])
        bad_logs += 1 if real else 0
    print("\nVERDICT (the numbers):", "every field of every JSON is identical, except the ones that "
          "are durations or timestamps" if worst == 0
          else f"{worst} file(s) DISAGREE on a number - see [DIFF] above")
    print("VERDICT (the logs):", "every line is identical, except durations, timestamps and one "
          "temporary directory name" if bad_logs == 0
          else f"{bad_logs} log(s) carry lines that differ by more than wall clock - read them: "
               "a header printed from an artifact that did not exist yet at that step is a real "
               "difference, and a real difference is the thing worth knowing about")
    return worst


if __name__ == "__main__":
    sys.exit(main())
