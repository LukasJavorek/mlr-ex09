# Review — agent-built tasks 1, 3, 6

For each task: the specification table as shipped, what was checked (smoke test, baseline,
predictions), and what the agent got wrong and how that was found. Every quoted number is in
the artifact named next to it; the commands in the last section reproduce them.

## Task 1 — ECG delineation (`task1_ludb/`)

**Specification table.** `spec.py:16-32`, printed at the top of every log before any number, with
an "AS BUILT" block read live from `results/choices.json`: input `(1,5000)`, output `(4,5000)` for
P/QRS/T/background, 1×1 conv with no activation, `CrossEntropyLoss(ignore_index=-1)`, margin 128 and
span-standardisation, class weights 0.5, `UNet(1,1,4,[32,64,128,256,512])`, Adam 1e-3, batch 16,
75 epochs, detector 8–20 Hz / 200 ms / 0.35 / p90. **No assert pins the table's prose to the code** —
unlike Task 6.

**What was checked.** `results/smoke_log.txt` (8 `[ok]` lines): cross-entropy equals a hand-computed
mean over the 14,594 annotated samples to 1.3e-07; Dice is exactly 1.0 from label-built logits; and
overwriting all 5,406 unannotated predictions leaves the loss **bit-identical** — the check that the
mask means what it says. Baseline = 8–20 Hz Butterworth → `find_peaks` → median P/QRS/T offsets
fitted on train: test Dice 0.738 / 0.881 / 0.829 against the network's 0.758 / 0.908 / 0.822. The
shown trace (`example_record.png`, record 121) is chosen by a written rule (`pick_record`, most
annotated QRS); re-derived from the cached arrays — 15 QRS is a unique maximum.

**What the agent got wrong.** Admitted in its own report: the first masking check tampered with
*targets* instead of predictions, so it proved nothing; a `/2·FS` unit error printed absurd wave
widths; a stage-A panel compared a tuple to a list and reported "48 settings" whose means were all
`nan`. Found in its artifacts: `baseline.py:33` enumerates 2×3×5×4 = **120** settings and then says
"96"; the same docstring names 5–15 Hz / median prominence / 240 ms where the chosen configuration is
8–20 Hz / p90 / **200 ms** (`choices.json`); `RESULTS.md:315` says the smoke test prints seven `[ok]`
lines — the log has eight; `results/` is gitignored and the `results/run_all_log.txt` the report
promises **does not exist**, yet it calls `results/` "the record" and says the per-rhythm table
survives *only* there; and `choices.json` contains a key `note_widths` **that no script writes** —
the artifact it calls authoritative was hand-edited. The audit's own unresolved records (138, and
120/126/127/133) are never cross-checked and never reach "Honest limits".

## Task 3 — spike inference (`task3_spikefinder/`)

**Specification table.** `spec.py:17-34`, copied from the notebook's filled cell, plus a live AS BUILT
block: `(1,512)` at 25 Hz, per-neuron z-scored → one rate per 40 ms bin, 1×1 conv + softplus, Poisson
NLL in units of the neuron's own mean rate, the challenge's own binned Pearson metric, both classical
baselines rebuilt with σ tuned on validation; window 128, widths `[16,32,64,128]`, lr 3e-3, batch 64,
24 epochs, duplicate `(9,3)` dropped. Again **no prose-to-code assert**.

**What was checked.** `smoke_test.py`, 9 checks / 28 asserts: the Poisson NLL identity to 1e-6; affine
invariance of the metric (worst deviation 0.0e+00, and a constant prediction returning `None` rather
than 0); a 30-window capacity fit reaching correlation 0.994; the split 122/18/34 → 121 after the
duplicate drop. Baseline on test: 0.206 (fluorescence level) and 0.467 (smoothed derivative) against
the model's 0.632. `test_scores.png` plots **all 34** held-out neurons individually rather than an
average, which is the single most useful figure in the folder.

**What the agent got wrong.** Admitted: the `(9,3)` train/test duplicate leak (found by the audit, then
re-measured with it put back), ≈0.026 optimism from selecting an epoch on 18 neurons, and a 90-epoch
collapse from 0.635 to 0.518. Found: `sf_data.py:270` prints "0–6 spikes per 40 ms" in **every log
header**, while `smoke_log.txt:9` two lines later says 0–15 observed; `RESULTS.md:263` claims
per-neuron standardisation is "worth **+0.002**" when its own `choices.json` scores `standardize none`
at 0.6199 and the standardised arm at 0.6176 — the sign is inverted; the ±0.045 noise floor used to
call ties appears in six places and is sourced in none (the audit says ±0.022, the folder's own
bootstrap on the same 18 neurons gives ±0.063). Most seriously, the mtimes contradict the stated order
of work: `results/model.pt` 13:13:53 and `train_log.txt` 13:13:54 **predate** `sf_data.py` 13:14:11 and
`train_task3.py` 13:14:24, so the shipped weights came from code edited afterwards;
`eval_test_log.txt` (13:22:35) precedes `eval_valid_log.txt` (13:28:24), so "the validation dry run came
before the single read of test" is not what the artifacts support; and `smoke_log.txt` (13:29:40)
postdates the trained model despite its "before any training" heading.

## Task 6 — colourisation (`task6_colorize/`)

**Specification table.** `spec.py` holds the notebook's cell-77 table *and* a `CONSTANTS` block that
**asserts** four of its stated numbers against the code (16×16 grid, 256 bins, ±128 chroma units,
40 dB clip), so drift fails the run — the only one of the three that enforces its own specification.

**What was checked.** 18 smoke checks, chiefly that the metric reproduces five numbers measured in
`task6_data_audit.md` **before the code existed** (valid gray 26.94 mean / 24.45 clipped / 21.06
pooled / 23.95 median / chroma L1 0.0804, all within 0.02); a zero-initialised head scoring the gray
baseline to exactly 0.00e+00, proving Y is copied back and the ±128 scale applied once; the
1024-of-4096 pixel loss being the same function as the full one (1.49e-07), which is what licenses one
of its own sweep arms; and both lookup ceilings (31.54 dB binned truth, 18.77% / 30.12%). Baseline gray
measured per split — 26.94 / 25.36 / 25.58 — not quoted. The seven photos shown were selected by rule
ET5 with the indices printed in `eval_test_log.txt`, and `metrics.py` re-runs the controls with no data
at all.

**What the agent got wrong — all of it disclosed, and all reproducible.** A call that could not work
(`figs.legend_on`) reached a trained model before the binding check existed; that check then caught its
own `figs.FIGURES`, i.e. the plotting path had never run. `pick`'s cost tie-break compared raw seconds,
so an arm would have won for training *fewer* epochs; R4's override path printed "the R4 pick is used"
while writing R3's knobs. Run #1's headers printed a rehearsal's `alpha 1.4` and "2 kept of 2 trained"
from artifacts left on disk — caught only by running the pipeline twice into an empty `results/` and
diffing, which showed every number identical. Two errors were in the reporting rather than the code: an
in-sample ceiling quoted against out-of-sample scores, and the validation α-cliff attributed to the
*test* split's colourless photos. `--report` prints route B's indifference band as 0.00458 where the
prose said 0.00464. And its headline result is a negative one it did not tune away: on the task's own
first metric, sending no colour wins (gray 25.36 vs 24.26), because ten near-black-and-white photos sit
on a 120 dB floor of the folder's own making. **Still open:** `spec.text()` fills "AS BUILT" from
whatever JSONs are on disk, so re-running into a dirty `results/` prints the previous run's
hyperparameters.

## Verdict

Task 6's results are accepted, because two independent 89-minute runs agree on every number and its
specification enforces itself. Task 1's model is accepted; its *record* is not — the authoritative
artifact was hand-edited and the console record it points to does not exist. Task 3's model is accepted
only because `eval_valid_log.txt` reproduces 0.6349 from the shipped weights; its provenance, its tie
threshold and one sign-flipped claim are wrong, and its order of work is contradicted by its own
timestamps.

## Reproduce

- `python smoke_test.py` in each of the three folders
- `python validate_choices.py --report` (tasks 1 and 6)
- `python spec.py` (task 6 — prints the four asserts)
- `stat` on the task-3 files quoted above, for the mtime ordering

Use `C:\dev\9\.venv\Scripts\python.exe`, run from inside the task folder. Nothing retrains.
`results/*.pt` and `figures/` are gitignored, and in task 1 all of `results/` is — ask for those or
re-run.
