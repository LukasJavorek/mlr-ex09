# Exercise 9 — network types: one encoder, many heads

Three of the exercise's tasks are coded, each in a folder of its own, each reproducible
from a fresh clone with one command. Everything shares the network code of Part 1 of
`ex09_student.ipynb` (`conv_block`, `Encoder`, `UNet`, `train`, `predict`) - those
classes are copied into each folder verbatim, and every task differs from the next
only in what it feeds them and what it asks them for. Task 6 is the fourth entry and a
different kind of thing: its data was audited and its specification table filled in, and
nothing in it was trained - see its section.

| # | task | data | what the network answers | folder | state |
| --- | --- | --- | --- | --- | --- |
| 1 | ECG delineation | LUDB, 200 × 10 s of lead II at 500 Hz, cardiologists' wave outlines | one of four classes — background, P, QRS, T — for every sample | [`task1_ludb/`](task1_ludb/) | **done** |
| 2 | Activity from the phone's accelerometer | UCI HAPT, 61 recordings of 3-axis acceleration at 50 Hz | one of seven classes (six activities + transitions) for every sample | `task2_hapt/` | not started |
| 3 | Calcium imaging → spikes | spikefinder, 174 neurons in 10 datasets at 100 Hz | a spike rate for every 40 ms bin (regression, not classes) | [`task3_spikefinder/`](task3_spikefinder/) | **done** |
| 6 | Colourisation of photos | Imagenette2-160, 13,394 photos as the notebook's data cell loads them (9469 train / 100 valid / 500 test) | the two chroma channels of a photo, from its own luminance | *(no folder)* | **audited + specified; nothing trained** |

The notebook itself is in this folder. It is **not** part of any task's code path:
`Copy_of_ex09_student.ipynb` was edited in exactly three places, the specification
tables under Task 1, Task 3 and Task 6, and each coded task's code lives in its own
folder instead.
Its 38 code cells are byte-identical to the originals - `_analysis/check_notebook.py`
compares them against a copy taken before the first edit.

## Dependencies

Python 3.10 or newer (everything here was tested on 3.13). One install per task:

```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r task1_ludb/requirements.txt            # numpy scipy pandas matplotlib wfdb torch
pip install -r task3_spikefinder/requirements.txt     # the same without wfdb
```

Both files pin the versions everything was measured on. The two tasks share nothing on
disk; running them in one environment is fine, and the pins are the same versions.

* **CPU is enough.** Task 1 takes about 2 h start to finish on a 6-thread laptop CPU
  and Task 3 about 1 h 30 m, and in both the bulk of that is the validation sweeps that
  choose the settings — the final models themselves train in 14 min and **88 s**.
  `--skip-choices` gets you from an existing `results/choices.json` to the
  trained model and its test numbers in ~20 min (Task 1) and ~2 min (Task 3). Nothing
  needs a GPU; on one the training runs would be faster, but not bit-identical
  (different kernels sum in a different order) — the setup, the seeds and the choices
  are the same either way.
* The data is downloaded by the scripts, from PhysioNet (Task 1), the UCI
  repository (Task 2) and the spikefinder challenge's own address (Task 3, 71 MB).
  Nothing needs an account; every folder keeps its download in its own `data/` and
  never adds it to git.

## How to run them

### Task 1 — ECG delineation (`task1_ludb/`)

Everything, in order, from an empty folder:

```bash
cd task1_ludb
pip install -r requirements.txt
python run_all.py                    # ~2 h on a CPU; downloads LUDB (15 MB) on first run

# the same, keeping the console output next to the results it produced: the scripts
# write the .json files themselves, the .txt logs in results/ are captured output
python run_all.py 2>&1 | tee results/run_all_log.txt
```

That is five steps, and each is also a script of its own if you want to run or
re-run one alone:

The times are what each step took on the 6-thread CPU everything here was measured on.
Where a number is in a file you can check it: the sweep's per-run seconds are in
`results/sweep_log.txt`, `refine_log.txt` and `confirm_log.txt`, and the training's
826.9 s is in `results/train_log.json`. The data, smoke-test and evaluation times are
wall-clock observations of those same scripts on the same machine, not logged figures.

| step | command | what it does | what it writes | took |
| --- | --- | --- | --- | --- |
| 0 | `python ludb_data.py` | downloads LUDB, reads all 200 records, **crops each to its annotated span** and prints what the data then looks like | `data/cache_*.npz` | ~1 min |
| 1 | `python smoke_test.py` | seven checks before any training: shapes, that the loss really skips the `-1` targets, that output channel *c* is class *c*, that what the network says where nobody labelled cannot change the score, and that two records overfit to Dice 0.99 | nothing, it only prints | ~2 min |
| 2 | `python validate_choices.py`, then `--refine`, then `--confirm` | every choice, made on the **validation** split: the baseline's detector thresholds, the crop and scaling, the class weights, the widths and the learning rate; then the epoch count, the batch size and the depth; then the depth again, measured at the length the run actually trains for | `results/choices.json`, `figures/validation_choices.png` | 16 + 36 + 54 min |
| 3 | `python train_task1.py` | trains the U-Net on the 140 training records with exactly those choices, keeping the best validation epoch | `results/model.pt`, `results/train_log.json`, `figures/training_curves.png` | 14 min |
| 4 | `python evaluate_task1.py` | the **one** look at the 30 test records, and the four things the task asks to show | `results/test_scores.json`, `figures/test_dice.png`, `figures/example_record.png` | ~2 min |

Shorter versions, both of which still train the same model and read the test split once:
`python run_all.py --skip-choices` (≈20 min — the four steps above with step 2 dropped,
reusing `results/choices.json`) and `python run_all.py --quick` (≈50 min: fewer settings
per sweep, and stage G's 150-epoch confirmation run skipped, which is the single most
expensive training in Task 1). One caveat on all of these totals: they are the pieces
added up, not a stopwatch on `run_all.py` — every stage was run and timed on its own, in
the order above, on one machine, and the individual figures are in the files named just
above.

What that ends up with, on the 30 test records — the task asks the U-Net to beat the
classical baseline on all three waves, and it does:

| test Dice | P | QRS | T |
| --- | --- | --- | --- |
| baseline (R peaks + median distances), measured here | 0.738 | 0.881 | 0.829 |
| **U-Net, this model** | **0.903** | **0.926** | **0.906** |
| the task text's baseline, for orientation, on its own label basis | 0.74 | 0.89 | 0.85 |

Results, and what was checked: [`task1_ludb/RESULTS.md`](task1_ludb/RESULTS.md).
The specification the code is built to: [`task1_ludb/spec.py`](task1_ludb/spec.py)
(`python spec.py` prints it, and `evaluate_task1.py` prints it before any number).

### Task 2 — Activity from the phone's accelerometer (`task2_hapt/`)

*Not implemented yet.* Planned shape, so that the structure above repeats:
the same `UNet(dim=1, ...)`, with `c_in=3` (the three acceleration axes) and
`c_out=7`; `CrossEntropyLoss(ignore_index=-1)` again, because the samples between
the labelled activities carry `-1` in this database too; per-recording windows
rather than whole recordings, since a recording is a few minutes long; and the same
`train` / `predict` pair. The exercise then runs the trained network in the browser
on the phone's own accelerometer, which needs the weights exported once
(`torch.onnx.export`, or the equivalent for the course app).

### Task 3 — Calcium imaging → spikes (`task3_spikefinder/`)

The exercise is a signal-to-signal regression rather than a classification, so
`ignore_index` and Dice both go away and the metric becomes the challenge's own: sum
the predicted rate and the true 10 ms counts into 40 ms bins, take a Pearson
correlation per neuron, average over the test neurons. The recordings differ in
length and each ends in NaN padding, so the loader must not see the padding; the
174 neurons differ in firing rate by a factor of 100, so a single pooled objective is
a different objective; and the calcium kernel is slow, so the network needs a longer
field of view than the released sampling rate gives it for free.

[`task3_data_audit.md`](task3_data_audit.md) and `task3_audit_figs/` measure all of
that before any model is written, and the specification table in the notebook is
filled in from those measurements. Three findings changed the modelling: dataset 9
contains **one neuron twice**, with the two copies in train and test; the score has a
**±0.005 noise floor** and only 18 validation neurons to tune on; and the calcium
kernel takes 0.4–2.3 s to fall to half its peak, longer than the U-Net's 0.44 s
receptive field unless the input is decimated.

Everything, in order, from an empty folder:

```bash
cd task3_spikefinder
pip install -r requirements.txt
python run_all.py                    # ~1 h 30 m on a CPU; downloads spikefinder (71 MB) first
python run_all.py 2>&1 | tee results/run_all_log.txt
```

The times are measured on the same 6-thread CPU, and for a change most of them are
*logged* rather than observed: the sweep prints its own stage seconds
(`17.4 / 44.2 / 25.2 min on cpu`, at the end of each of `sweep_log.txt`,
`refine_log.txt` and `confirm_log.txt`) and `train_log.json` holds the training's 88.1 s.
The loader, smoke-test and evaluation seconds are wall-clock of those same scripts.

| step | command | what it does | what it writes | took |
| --- | --- | --- | --- | --- |
| 0 | `python sf_data.py` | downloads spikefinder, reads all 174 neurons, decimates 100 Hz → 25 Hz, standardizes each neuron from its own trace, rebuilds the notebook's split and prints what is left | `data/cache_*.npz` | seconds from the cache; the first run downloads 71 MB and reads 20 csv |
| 1 | `python smoke_test.py` | nine checks: the grid and the standardization are what they claim, no window contains a NaN, the split is the notebook's with the duplicate out, the shapes round-trip, the Poisson head equals the textbook NLL, the metric is affine invariant and returns `None` for a constant, the controls sit at zero, 32 windows overfit to 0.994, and two identical runs agree | nothing, it only prints (`results/smoke_log.txt`) | **6.5 s** |
| 2 | `python validate_choices.py`, then `--refine`, then `--confirm` | every choice, made on the **validation** split, three seeds per arm: the sampling grid, window, standardization and target scale; then lr, depth, batch and the epoch count off a 90-epoch curve; then the winner at full length beside the runner-up, beside the reopened leak, and beside the arm that separates a short window's two advantages | `results/choices.json`, `figures/validation_choices.png`, `figures/validation_curves.png` | 17.4 + 44.2 + 25.2 min |
| 3 | `python train_task3.py` | trains the U-Net on the 121 training neurons with exactly those choices, keeping the best validation-**correlation** epoch (not the best-loss one — the two disagree, and `figures/training_curves.png` shows both) | `results/model.pt`, `results/train_log.json`, `figures/training_curves.png` | **88 s** |
| 4 | `python evaluate_task3.py` | the **one** look at the 34 test neurons, and everything the task asks to show | `results/test_scores.json`, `figures/test_scores.png`, `figures/example_neuron.png` | ~6 s |

`python evaluate_task3.py --valid` runs that last script's code path end to end on the
validation split first, writing `test_scores_valid.json` and `…_valid.png`; plotting
code that has never been executed is where a final evaluation fails, and failing
halfway through would mean the test split had been loaded for nothing.
`python baseline.py` re-derives the two classical baselines and their σ tuning on its
own, on train and valid; it can score the test split but only when you pass `--test`, so
"step 4 is the one look at it" stays true of an ordinary pass through the folder.

Shorter versions: `python run_all.py --skip-choices` (≈ **2 min**, reusing
`results/choices.json` — or start from `choices.json.example` to train the specification
as written without any sweep at all; note that file trains the spec *as written*, not the
lr, depth, batch and epoch count the sweep measured) and `python run_all.py --quick`
(≈ 15 min: one seed per arm, 8-epoch arms, no stage 3 — enough to see the pipeline work,
not enough to rank settings, since one seed cannot separate differences under the ±0.045
noise of 18 validation neurons. This one figure is extrapolated from the measured
full-stage seconds rather than stopwatched: `--quick` was never run end to end, because
running it would overwrite the `results/choices.json` the shipped numbers come from).
As in Task 1, the totals are the pieces added up on one machine,
not a stopwatch on `run_all.py`; the per-run seconds are in
`results/sweep_log.txt`, `refine_log.txt`, `confirm_log.txt`, `train_log.json` and
`baseline_log.txt`, and `run_all.py`'s own header quotes the same measured stages.

What that ends up with, on the 34 test neurons (the challenge's metric — correlation
with the true spike counts, mean over neurons):

| test correlation | mean over neurons | mean over the 10 datasets |
| --- | --- | --- |
| fluorescence, the signal itself | 0.206 [+0.172, +0.240] | 0.222 |
| smoothed positive derivative, σ tuned on validation | 0.467 [+0.416, +0.523] | 0.445 |
| **U-Net, this model** | **0.595 [+0.544, +0.647]** | **0.572** |

**The task's target is met on both readings:** +0.128 over the derivative on the mean
(7.2 paired bootstrap sd, against a ±0.005 noise floor on a mean of 34 neurons), and in
**9 of the 10 datasets** — the exception being dataset 3, by 0.013, on three neurons.
The same two baselines rebuilt in the audit score 0.206 and 0.467 on this split, which is
the same measurement made twice, and the task text quotes 0.21 and 0.45 for them. The
controls sit where they must: the same predictions rolled through time score −0.001, and
against another neuron's spikes +0.005.

Results, and what was checked: [`task3_spikefinder/RESULTS.md`](task3_spikefinder/RESULTS.md).
The specification the code is built to: [`task3_spikefinder/spec.py`](task3_spikefinder/spec.py)
(`python spec.py` prints the table plus the settings the final run actually used, and
`evaluate_task3.py` prints it before any number).

### Task 6 — Colourisation of photos (no folder: `task6_data_audit.md`)

Audited and specified, **not trained** — the deliverable here is the data audit and the
specification table in the notebook (cell 77), both filled in from measurements rather
than from guesses. It needs no install beyond Task 3's requirements and no code of its
own: the four audit scripts live in `_analysis/` and take the data directory, so the
whole thing reproduces with

```bash
python -m venv .venv && . .venv\Scripts\activate && pip install -r task3_spikefinder/requirements.txt

# the data, a 94 MB tgz, the same folder the notebook's data cell downloads and untars
curl -LO https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz && tar -xzf imagenette2-160.tgz
DATA=imagenette2-160              # any folder holding train/ and val/; here: _imagenette_dl/imagenette2-160

python -I _analysis/col_inventory.py $DATA  > _analysis/col_inventory_log.txt   # builds the cache: decodes all 13,394
python -I _analysis/col_dups.py      $DATA  > _analysis/col_dups_log.txt       # 14 s from the cache
python -I _analysis/col_oracle.py    $DATA  > _analysis/col_oracle_log.txt     # 12 s, + _analysis/col_oracle/oracle.json
python -I _analysis/col_figs.py      $DATA  > _analysis/col_figs_log.txt       # 18 s, + task6_audit_figs/*.png
```

The four scripts write the `.json`/`.png` files beside themselves and print the tables, so
the `.txt` logs above are captured stdout. Each one fills `_analysis/col_cache/` on its
first run and reuses it after; that cache was checked against the files (re-running
`load_photo` on 400 photos reproduces the cached bytes exactly), so the audit is computed on
the same pixels the notebook will train on.

Everything is fitted on the 9469 training photos and decided on the 100 validation
photos; the 500 test photos were read only to re-check the two numbers the task text
quotes for them (the gray baseline's ~25 dB, the photos' ~0.04 colourfulness). What the
measurements changed about the task:

* **The task's own first metric is maximised by refusing to colour.** Shrinking the
  L2-optimal answer by a factor α (α = 0 is gray, α = 1 is the conditional mean) gives a
  per-photo-then-mean PSNR that *peaks at α = 0*, 26.94 dB, and falls from there — while
  the pooled PSNR the notebook's own cell 30 computes peaks at α = 1.30. The two
  conventions in use rank the predictors in opposite directions, so the eval metric in the
  spec table is three numbers with their convention written down, not one.
* **Both routes have low, now-measurable ceilings.** The luminance carries 1% of the
  colour: the conditional mean is the ceiling of route A at 24.36 dB / colourfulness 0.011
  — that washed-out ratio *is* the task's "dull brownish gray", measured before any
  network exists. Route B's ceiling depends on the bin count, which is why the spec says
  256 bins: a 36-colour vocabulary caps the PSNR at 22.3 dB even with a perfect classifier,
  which on the task's per-photo convention is *below* the gray baseline — and 22.0 against
  gray's 21.1 on the pooled one, so the convention is half of why that is a trap. Best
  possible top-1 accuracy on a 32×32 chroma grid is **18.8%**, so that metric is quoted
  against its ceiling and never as a headline.
* **The published prior-correction is off by a scale factor.** γ = 1 on unnormalised CbCr
  paints every pixel at the edge of the colour space (colourfulness 0.236, 7.9 dB);
  γ = 0.3 was measured on valid, and belongs at inference only.
* **Three things the task text does not mention exist.** 3 of the 100 valid photos and 8
  of the 500 test photos are *already* black and white, which the gray baseline reproduces
  exactly, so the baseline's PSNR is a property of the epsilon in the code: on the same 500
  test photos it is 25.36 dB with the `1e-12` floor the code uses, 24.40 dB capped at 60,
  24.06 dB capped at 40 and 23.82 dB with those 8 photos left out — median 23.21 dB, which
  no convention moves. 3325 of the 3925 validation photos are used by nobody, next to a
  ±1.7 dB noise floor on the 100 that are. And the loader is itself a choice with a number
  on it: the centre crop keeps a median 75% of the resized photo (minimum 20%, because
  aspect ratios reach 5:1), and the un-prefiltered BICUBIC 160→64 shrink disagrees with a
  properly-filtered LANCZOS by 1.24 of 255 per pixel — a 42.8 dB floor under how much fine
  colour detail any model here can show, the same order as the 25 dB the task argues about.

The full audit — ten sections, six figures, and a "checked and clean" list so nobody
re-litigates the EXIF flag or the split's file-name collisions:
[`task6_data_audit.md`](task6_data_audit.md), `task6_audit_figs/`, raw numbers in
`_analysis/col_inventory_log.txt`, `_analysis/col_dups_log.txt`,
`_analysis/col_oracle_log.txt`. The specification the notebook now carries is cell 77 of
`Copy_of_ex09_student.ipynb`, two columns (A — regression, B/B+ — bins) because the task's
"what to show" asks for both.

## The rules every task here keeps to

* The **validation** split decides every choice — preprocessing, hyperparameters,
  thresholds, how long to train. The **test** split is read once, by the last
  script, and its numbers are never used to decide anything.
* Every script states which splits it reads. `evaluate_task1.py` and
  `evaluate_task3.py` are the only files in their folders that score the test split,
  Task 3's `baseline.py` needs an explicit `--test` to do so, and Task 3's sweeps do
  not even materialize its windows (`sf_data.sets(data, ("train", "valid"))`, with an
  assert that none appeared).
* Anything that is measured from data (the baseline's median wave distances, the
  per-neuron target scale) is measured on the **training** records only. Task 3
  asserts it: `validate_choices.start()` refuses to run if any test neuron has a
  target scale.
* A baseline is rebuilt on the given split, not quoted from the paper, so that the
  comparison table is one number over the same bins of the same neurons.
* A score is never quoted without its noise. Both tasks measure how far the metric
  moves on luck — ±0.005 for Task 3's mean over 34 neurons — and both report controls
  (predictions rolled through time, scored against another neuron's labels) that must
  sit at zero.
* Every figure is saved as a `.png` and every number is printed and written to a
  `.json`, so a run is auditable without opening a single plot.

## Layout

```
README.md                      this file
Copy_of_ex09_student.ipynb     the exercise notebook (its Task 1, Task 3 and Task 6 spec tables filled in)
task1_ludb/
  ludb_data.py                 the loader: LUDB -> Option B crop -> arrays -> torch datasets
  part1.py                     the notebook's conv_block / Encoder / UNet / train / predict, verbatim
  spec.py                      the specification table, as printed text
  baseline.py                  the classical delineator the task defines
  metrics.py                   Dice over the annotated samples, and the QRS durations
  smoke_test.py                the seven checks to run before training
  validate_choices.py          every choice, made on the validation split
  train_task1.py               the final training run
  evaluate_task1.py            the one test evaluation, and the four things to show
  run_all.py                   all of it in order
  RESULTS.md                   results, and what was verified
  requirements.txt
  data/ results/ figures/      generated, not in git
task2_hapt/                    (to be written)
task3_spikefinder/
  sf_data.py                   the loader: spikefinder -> 25 Hz -> arrays -> torch datasets
  part1.py                     the notebook's conv_block / Encoder / UNet / train / predict, verbatim
  spec.py                      the specification table, as printed text
  baseline.py                  the fluorescence and smoothed-derivative baselines, tuned on validation
  metrics.py                   the challenge's correlation, its bootstrap and its two controls
  fitting.py                   the softplus head, the Poisson loss, and a train loop that selects on the correlation
  figs.py                      the one plotting style (shared with the audit figures)
  smoke_test.py                the nine checks to run before training
  validate_choices.py          every choice, made on the validation split, in three stages
  train_task3.py               the final training run
  evaluate_task3.py            the one test evaluation (and a --valid dry run of it)
  run_all.py                   all of it in order
  RESULTS.md                   results, and what was verified
  requirements.txt             and choices.json.example, to train the spec without the sweeps
  data/ results/ figures/      generated, not in git
_analysis/                     the scratch scripts the audits and this README were built with
  col_inventory.py             Task 6: what the data cell loads, section by section
  col_dups.py                  Task 6: near-duplicate photos, across and inside the splits
  col_oracle.py                Task 6: every ceiling, no network (the α dial, the bin counts)
  col_figs.py                  Task 6: the six figures
task1_data_audit.md            task3_data_audit.md       the data audits, written before the code
task1_audit_figs/              task3_audit_figs/         their figures
task6_data_audit.md            the Task 6 audit, written before any model: every ceiling, every
                               trap, a solution per section, and the source of its spec table
task6_audit_figs/              its six figures
```
