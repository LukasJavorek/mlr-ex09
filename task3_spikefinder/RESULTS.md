# Task 3 — Calcium imaging → spikes: results, and what was checked

Every number here comes out of a script in this folder and every figure is in
`figures/`. The record is in `results/` at two levels: the `.json` files are written by
the scripts and are the authoritative ones, the `.txt` files are the console output of
those same runs. `python run_all.py` reproduces the whole chain from an empty folder;
`python spec.py` prints the specification the code is built to, and every script that
produces a number prints it first, so a specification and a result cannot quietly drift
apart. One provenance note up front: the shipped `results/` came from **staged** runs —
each stage was timed and logged separately, in this order, on one machine — so where
this report and `README.md` quote a total it is the measured stages added up, not one
stopwatch on `run_all.py`.

## The bottom line

Task 3 asks for a spike **rate**, sample by sample, from the fluorescence, scored by the
challenge's own metric (sum both sides into 40 ms bins, Pearson per test neuron, mean
over neurons), against two classical guesses the task text puts at about **0.21** (the
fluorescence itself) and about **0.45** (the smoothed positive derivative) — and it asks
the network to beat the derivative *on average and in most of the ten datasets*.

| test split, 34 neurons, 2.7 h of recording, 245 thousand 40 ms bins | mean over neurons [bootstrap 95%] | mean over the 10 datasets | per-neuron range |
| --- | --- | --- | --- |
| fluorescence (*task text: "about 0.21"*) | 0.206 [+0.172, +0.240] | 0.222 | +0.001 .. +0.458 |
| smoothed positive derivative, σ tuned on validation (*"about 0.45"*) | 0.467 [+0.416, +0.523] | 0.445 | +0.150 .. +0.762 |
| **U-Net, this model** | **0.595 [+0.544, +0.647]** | **0.572** | +0.255 .. +0.798 |
| margin over the derivative | **+0.128**, paired bootstrap sd 0.018 → **7.2 sd** | +0.127 | — |

**It beats it, on both readings of the target.** On the two means, by +0.128 and +0.127;
and in **9 of the 10 datasets** (the exception, dataset 3, by 0.013 — read against the
±0.03 that a *single* neuron is worth, and against that dataset's mean resting on three
neurons). Both baselines were re-measured on this split rather than quoted, and both land
on the task text's numbers, so the bar is the exercise's own bar.

Three things printed beside those rows, because a mean over correlations is easy to
over-read:

* **controls.** The same 34 predictions rolled through time score **−0.001**; scored
  against *another* neuron's spike train, **+0.005**. The score is not the signals' own
  autocorrelation being counted as success.
* **nothing is undefined**: 0 constant predictions out of 34, for all three methods. A
  network that answered with a flat line would have shown up as a missing neuron, not a
  low number.
* **the noise floor of the mean itself.** One neuron is worth about ±0.03 (measured in
  the audit by scoring the two halves of every recording against each other), so the
  mean over 34 moves ~**±0.005** on luck alone. The gap is 26 of those; the loss in
  dataset 3 is not of that kind of size at all.

On the **validation** split, where every choice was made, the same saved model scores
**0.635** against the derivative's 0.455 and the fluorescence's 0.191, beating the
derivative in **10 of 10** datasets there — the test result is the same gap on a split
nothing was tuned on, arrived at by a different set of neurons.

Model: `UNet(dim=1, c_in=1, c_out=1, widths=[16, 32, 64, 128])`, **169,713 parameters**,
input `(1, 128)` at 25 Hz (5.1 s), output `(1, 128)` — one rate per 40 ms bin — trained
24 epochs with **epoch 11** (the best validation-correlation epoch) kept, **88 s** on a
6-thread CPU.

## The two baselines, quoted and re-measured

The derivative's smoothing width is the one knob the baseline has, and the task says to
choose it on the validation neurons. `python baseline.py` does exactly that
(`results/baseline_log.txt`):

| σ | 10 ms | 20 ms | 30 ms | **50 ms** | 80 ms | 120 ms | 200 ms | 300 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation correlation | 0.363 | 0.415 | 0.439 | **0.455** | 0.454 | 0.438 | 0.396 | 0.352 |

The whole curve spans 0.103 but its **top two widths differ by 0.0003**, and the audit
re-tuned σ on bootstrap resamples of those 18 neurons with the winner changing in a third
of the draws — so σ is worth choosing and is not worth arguing about. It is the best of
the grid, so this is not a crippled baseline.

Both baselines are then stable across the three splits, which is a useful check that the
notebook's per-neuron split produced three exchangeable sets of neurons and nothing else:

| | train (121) | valid (18) | test (34) |
| --- | --- | --- | --- |
| fluorescence | 0.191 | 0.191 | 0.206 |
| derivative, σ = 50 ms | 0.460 | 0.455 | 0.467 |

One dataset deserves naming: **dataset 4 is where the derivative is not even the better
of the two baselines** (0.259 against the fluorescence's 0.281) — the audit's slow-kernel
case, where differentiating a calcium transient that takes seconds to fall mostly
amplifies noise. The task's target is "beat the derivative", and on that dataset the
thing to beat is the easier of its two baselines; the network beats both there
(0.549, i.e. +0.290 over the derivative). `evaluate_task3.py` marks such rows with a `*`
rather than quietly keeping them in an average.

## The three things the task asks to show

### 1 · The specification

`python spec.py` prints the notebook's six-row table, then an **AS BUILT** block read
from `results/choices.json`, so the promised specification and the model that produced
the numbers are the same object:

| | answer |
| --- | --- |
| network input (shape) | `(1, 128)` — one neuron's fluorescence, 100 Hz → 25 Hz by averaging every 4 samples, z-scored on that neuron's own trace; 128 samples = 5.1 s; windows cut at the end of the real part so none contains NaN padding |
| network output (shape) | `(1, 128)` — the predicted spike rate for the same window at the same 25 Hz: one rate per 40 ms bin, the metric's own resolution; fully convolutional, nothing cropped |
| last layer | 1 × 1 convolution to 1 channel, then **softplus** — non-negative, and still with a gradient where 95.6% of the bins sit (ReLU would be dead exactly at 0) |
| training loss | Poisson NLL `ρ − r·log ρ` on the 40 ms counts, with the rate in units of **that neuron's own mean rate**; per-window mean, then per-batch mean |
| evaluation metric | the challenge's own, unchanged; reported beside the mean over the ten datasets, the per-dataset means, the count of undefined predictions and the ±0.005 noise floor |
| baseline | both classical guesses, rebuilt on this split and scored by the same function; σ tuned on the validation neurons only |

*AS BUILT:* 4-sample mean to 25 Hz · window 128 = 5.1 s · per-neuron statistics ·
duplicate dropped · counts per 40 ms bin in units of the neuron's own mean rate ·
`widths=[16, 32, 64, 128]` + softplus · Poisson NLL · Adam lr 3e-3, batch 64 · 24 epochs,
best validation-correlation epoch kept · derivative baseline σ = 50 ms · seed 0.

Three things the table cannot hold, because the audit of the data
(`../task3_data_audit.md`) found them and the code has to say what it does about them:
dataset 9 contains one neuron **twice** (columns 3 and 4 are the same recording; the
notebook's split puts one copy in train and one in test), so `(9, 3)` is dropped from
training; the model is selected on the validation **correlation**, not on the loss
(`part1.train` early-stops on the loss, and the two peak in different epochs here); and
nothing is tuned per dataset, because five of the ten datasets have exactly **one**
validation neuron.

### 2 · The mean test correlation, and the same per dataset

The two tables at the top of this file, printed by `evaluate_task3.py`
(`results/eval_test_log.txt`) and drawn as `figures/test_scores.png` — whose right panel
plots all 34 neurons individually, because this metric is a mean over a short, wide list
and the list is the finding.

| dataset | test neurons | network | derivative | fluorescence | worst neuron | |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0.511 | 0.459 | 0.191 | 0.488 | |
| 2 | 4 | 0.420 | 0.307 | 0.198 | 0.255 | |
| 3 | 3 | 0.464 | **0.477** | 0.215 | 0.297 | the one loss, by 0.013 |
| 4 | 1 | 0.549 | 0.259 | 0.281 | 0.549 | `*` derivative < fluorescence |
| 5 | 2 | 0.432 | 0.332 | 0.328 | 0.347 | the hardest dataset |
| 6 | 2 | 0.668 | 0.578 | 0.315 | 0.660 | |
| 7 | 7 | 0.662 | 0.476 | 0.183 | 0.409 | |
| 8 | 4 | 0.745 | 0.554 | 0.104 | 0.714 | |
| 9 | 4 | 0.562 | 0.369 | 0.109 | 0.457 | holds the duplicate |
| 10 | 5 | 0.706 | 0.641 | 0.297 | 0.362 | |
| **mean over the ten datasets** | | **0.572** | 0.445 | 0.222 | | |

The mean over *datasets* (0.572) and the mean over *neurons* (0.595) are different
numbers and both are reported: the challenge weights neurons, so a dataset with 7 test
neurons counts seven times as much as dataset 4's one. The spread across datasets
(0.42 → 0.75) is wider than the gap over the baseline, which is the honest headline:
this is one network doing a decent job on ten different labs and indicators, not a
method with a measured quality.

**The duplicated neuron.** Test neuron `(9, 4)` — the recording whose training copy this
pipeline drops — scores 0.642. Its exclusion changes the mean over neurons from 0.595 to
**0.594**, i.e. not at the third decimal. Both numbers come out of the single read and
neither was chosen between; the run that left `(9, 3)` in training scored 0.626 against
the shipped setting's 0.632 on *validation*, so the leak was not buying performance
either way and the drop is a correctness fix, not a score fix.

### 3 · Thirty seconds of one test neuron

`figures/example_neuron.png` — neuron **3,4**, which scores 0.654, the median of the
split, and the **first** 30 s of its 270 s recording. Both the neuron and the window are
fixed by rule before anyone looks: `example()` takes the neuron whose score sits nearest
the median of the split (typical, not flattering) and always starts at t = 0.

Both continuous traces are rescaled to their own range and labelled as such, because the
metric is affine invariant — the shape is the whole content and the absolute rate is not
visible to it. What the picture shows is what the 0.595 is made of: the network's output
is a set of narrow transients that land on the spike bursts, and it goes flat across the
silent 12–25 s stretch, while the fluorescence that feeds it is still elevated for
seconds after the 2 s burst — which is precisely why the fluorescence itself only scores
0.21. The derivative (dashed, bottom) moves where the network moves at the 2 s and
27 s bursts and misses in between, which is most of the +0.128. The misses are visible
too: the network predicts nothing for the isolated spikes at 5, 11 and 17.5 s.

## What was checked before any training: `python smoke_test.py`

**Nine checks, 6.5 s, each printing a line and asserting** (`results/smoke_log.txt`):

| # | check | what it proves |
| --- | --- | --- |
| 1 | data | 174 neurons; x standardized per neuron (worst \|mean\| 1.6e-07, worst \|std−1\| 3.5e-06); y integer counts per 40 ms bin, 0–15 observed against a ceiling of 24, 95.6% empty, mean 0.062/bin; target scales exist for the 139 train+valid neurons and **for no test neuron** |
| 2 | no padding | every one of the 9957 windows of 128 samples ends at or before its neuron's last real sample, checked against 174 raw columns re-read from the csv through a second code path; 533 s of the 51513 s recorded — the ragged tail under one window per neuron — is all the signal no window sees |
| 3 | split | the notebook's `rng(0)` split rebuilt to the neuron: 122/18/34 with per-dataset `n_test = max(1, round(0.2n))`; dataset 9 really did hold the leak — `(9,3)` trained on while `(9,4)` sits in test — and with it dropped the 121 training neurons share no recording with any of the 34 test neurons |
| 4 | shapes | `(4, 1, 128)` in → `(4, 1, 128)` out, one non-negative rate per 40 ms bin, nothing cropped; a whole recording of 17996 samples (not a multiple of 8) comes out 17996 samples, so a test neuron is one forward pass and no stitching of overlapping windows |
| 5 | loss | `poisson_nll(ρ, r)` = 1.50493 equals the textbook `mean(λ − y·log λ)` of the real counts divided by that neuron's μ plus `mean(r)·log μ` to 1e-6: one likelihood, renormalized per neuron |
| 6 | metric | a perfect prediction scores 1.0 and so does every positive affine map of it (×2, ×2 as float32, ×1e-6, /μ, −mean, +1e6), the worst deviating by 0.0e+00; a constant prediction is `None`, not 0; untrained predictions rolled through time score −0.004 (another neuron: +0.001, above zero only because same-session neurons share signal) |
| 7 | capacity | 32 windows, 120 batches: loss 0.564 → −2.682 and correlation **0.994** on those same windows, over the 30 that carry a spike |
| 8 | determinism | the same batch twice gives bit-identical outputs in eval mode |
| 9 | head | with the 1 × 1 head driven to logits of −225 the output stays in [0, 527.3] and finite: softplus is non-negative everywhere and still has a gradient at 0 |

Two things worth naming, because they are the reason the checks exist:

* **Check 7 died on its first unloaded run**, with `TypeError: unsupported operand type(s)
  for +: 'NoneType' and 'float'`. The cause was not the network: once the window was the
  chosen **128** samples rather than the dataclass default of 512, **27% of the training
  windows contain no spike at all**, and a window whose target is empty has no correlation
  to score — the metric's own `None` rule, arriving inside a test that had been written
  when windows were long enough never to hit it. The check now asserts that *exactly* the
  all-empty windows are the unscored ones (so a genuinely flat network output would still
  fail it) and reports how many were excluded. This is a real property of the data at this
  window length, not a test artifact, and it is now printed instead of crashing.
* The two runs of the sweep and the final model are **bit-reproducible across processes**:
  a stage-2 run that was killed mid-flight (when the depth rule changed — see below) left
  `results/refine_log_stopped_at_5levels.txt`, and the relaunched run reproduced its arms
  to the third decimal, with only the seconds moving (169 → 174 s). That killed log is
  kept rather than deleted for exactly that reason.

## Every choice, made on the validation split

`python validate_choices.py` (stage 1), `--refine` (stage 2), `--confirm` (stage 3).
`figures/validation_choices.png` plots every arm including the losers, with true
seed-to-seed whiskers; `figures/validation_curves.png` shows the long curves;
`results/choices.json` holds every arm's per-seed scores, its curve and its measured
seconds. The test split is not read by any of them — `validate_choices.start()` asserts
that no test window is even materialized, and the sweep asks `sf_data.sets()` for
`("train", "valid")` only.

Arms are 3 seeds × 16 epochs unless stated; the spread quoted is the **sd over those three
seeds** on the 18 validation neurons, whose own bootstrap interval is ±0.045 — so an arm
difference under about 0.02 is a statement about noise.

**Stage 1 — the data** (16 epochs × 3 seeds, all arms `widths=[16,32,64]`, lr 3e-3, batch
32; 17.4 min):

| arm | validation correlation | sd over seeds | seconds |
| --- | --- | --- | --- |
| 100 Hz grid, window 2050 (20.5 s) | 0.609 | 0.005 | 301 |
| 50 Hz grid, window 1025 (20.5 s) | 0.620 | 0.004 | 136 |
| 25 Hz grid, window 512 (20.5 s) | 0.618 | 0.005 | 80 |
| **25 Hz grid, window 128 (5.1 s)** | **0.628** | 0.001 | 182 |
| 25 Hz grid, window 256 (10.2 s) | 0.624 | 0.002 | 117 |
| 25 Hz grid, window 1024 (41 s) | 0.612 | 0.003 | 67 |
| no per-neuron standardization | 0.620 | 0.002 | 79 |
| one global target scale instead of per-neuron | 0.579 | 0.007 | 78 |

**Stage 2 — the optimizer** (window 128; 44.2 min): lr 1e-3 → 0.620 ±0.006, **lr 3e-3 →
0.628 ±0.001**, lr 1e-2 → 0.621 ±0.001 · 3 levels 0.628 ±0.001 (0.042 M, 173 s),
**4 levels 0.630 ±0.003** (0.170 M, 262 s), 5 levels 0.634 ±0.004 (0.680 M, 419 s) ·
batch 16 → 0.629 ±0.002 (450 s), batch 32 → 0.630 ±0.003 (276 s), **batch 64 → 0.632
±0.004** (191 s) · and one 90-epoch curve at the chosen setting, which is the next section.

**Stage 3 — the winner at full length** (24 epochs × 3 seeds; 25.2 min):

| arm | correlation | sd | seconds |
| --- | --- | --- | --- |
| **the chosen setting** | **0.632** | 0.004 | 267 |
| runner-up: 5 levels, at its own best batch | 0.634 | 0.004 | 602 |
| duplicate `(9, 3)` put back into training | 0.626 | 0.002 | 275 |
| window 128 with batch 256 (≈ the same steps as the next row) | 0.615 | 0.005 | 192 |
| window 512 with the chosen batch 64 | 0.617 | 0.004 | 169 |

These full-length runs span 0.019 — inside the ±0.045 that 18 neurons cannot resolve — so
what stage 3 does is **rule settings out**, not rank them, and it says so on stdout.

Five findings, and the rule that turned them into a configuration:

* **Decimating to 25 Hz is free.** At equal context (20.5 s), equal window count and equal
  gradient steps, 25 Hz scores 0.618 against 100 Hz's 0.609 — nothing measurable is lost,
  each step costs about an eighth of the compute, and the same depths get a **4× larger
  receptive field**, which is what the calcium kernel actually needs.
* **The per-neuron target scale is the biggest number in the sweep: +0.039** (0.618 →
  0.579 when one global rate replaces it). That is the audit's weighting argument
  measured: an unweighted loss over concatenated windows gives 10 of the 174 neurons 30%
  of its weight, while the score weights all 174 the same.
* **Per-neuron standardization is worth +0.002 — inside the noise.** It is kept because
  it costs nothing and it is what makes BatchNorm see one distribution, but it is
  recorded here as *not* a measured win. Choosing on a 0.002 difference over 18 neurons
  would be choosing on noise.
* **The short window's advantage was gradient steps, not context — and stage 3 is what
  caught that.** Stage 1 made window 128 look 0.010 better than window 512, but a
  128-sample window also yields 4× as many windows, hence 4× as many updates for the same
  number of epochs: the two advantages are tangled. Stage 3 untangled them with three arms
  that all train for the same 24 epochs:

  | arm (all 24 epochs) | updates | windows seen | correlation |
  | --- | --- | --- | --- |
  | window 128, batch 64 — the shipped setting | 2640 | 167,832 | **0.632** |
  | window 128, batch 256 | 672 | 167,832 | 0.615 |
  | window 512, batch 64 | 648 | 40,680 | 0.617 |

  (updates = epochs × ⌈windows/batch⌉, so the last partial batch counts; the two
  648/672 arms differ by 4%, not by nothing.)
  Quadrupling the **updates** on identical data buys **+0.017** (rows 1–2). Quadrupling
  the **data** at the same number of updates buys **+0.002** (rows 2–3) — nothing. So neither the
  context length nor the amount of signal seen matters here; what mattered was optimizer
  steps, and the short window is honestly described as a cheap way to buy them, not as
  evidence that 5.1 s is the right context. What *is* measured about the receptive field —
  the quantity the context actually controls — gets its own argument below, and it is a
  separate one.
* **Depth was decided by a measured receptive field, not by the argmax.** 5 levels won the
  sweep by +0.004, which is inside noise, and would have driven both the 90-epoch curve
  and the shipped model. The receptive fields were measured instead of assumed — 44
  samples at 3 levels, 92 at 4, 188 at 5 — i.e. at 25 Hz: **1.76 s, 3.68 s, 7.52 s**. The
  audit measured the fluorescence taking 0.4–2.3 s to fall back to half its peak, so 3
  levels is too myopic, 4 covers the slowest kernel, and 5 sees 7.5 s into a 5.1 s window
  while costing 2.3× the compute and 4× the parameters. Stage 3 ran both at full length:
  0.632 against 0.634, the same number. 4 levels kept.

**The tie-break rule, written down because "the code chose it" should mean something
specific** (`validate_choices.pick()`, re-applied to every sweep by `rechoose()` before
anything is saved): arms that exist to test a hypothesis (`standardize none`,
`target scale`) can never be *picked*; among arms within **TIE = 0.005** of the best —
the noise floor of a mean over 18 neurons, not an arbitrary roundness — lr and batch are
decided by the **cheapest** arm measured in real seconds, and depth by the receptive-field
constraint `kernel half-decay ≤ RF ≤ window`. Every one of those three rules changes at
least one shipped number relative to plain argmax, which is why it is in the code and not
only in this file.

## The epoch budget, and what selecting on validation costs

The 90-epoch curve at the chosen setting (`figures/validation_curves.png`) peaks at
**epoch 11 with 0.635** and then falls — not gracefully: by epoch 90 it is at 0.518, below
epoch 3. Meanwhile the *training* loss keeps improving (−0.46 → −1.39 over 24 epochs), so
a loop selecting on loss would train past the peak and keep a worse model: at epoch 24 the
model is **better by the loss and worse by 0.022 on the metric**. Hence 24 epochs with the
best-validation-correlation epoch kept.

That selection is not free and the size of the discount is measured rather than waved at:
the sd of the curve's values near its own peak is **0.015**, and taking the maximum of
~24 such draws is worth about 1.7 sd, i.e. **≈ 0.026 of optimism**. The reported test
number sits 0.040 below the validation peak, which is what that discount plus
split-to-split noise (each mean carries a ±0.05 interval) predicts. The same figure shows
both facts on purpose: the peak ring, and the collapse after it.

## The final run

`python train_task3.py`: the 121 training neurons, 6993 windows of 5.1 s, Adam lr 3e-3
batch 64, `seed_everything(0)`, 24 epochs in **88 s**, keeping **epoch 11** by validation
correlation (0.635) where the validation loss alone would have kept epoch 13 (0.633) —
the two criteria agree here by luck, not by construction, and `figures/training_curves.png`
plots both curves plus the training loss so a reader can see how close that call was.

The saved model scores 0.635 on validation and, in a **separate process** reading only
`results/model.pt` and `results/choices.json`, the evaluator reproduces that number to
four decimals (0.6349) — which is the check that the shipped weights are the weights that
produced the number.

One more number worth having: the same model scores **0.593 on the 121 neurons it trained
on**, essentially the same as the 0.595 it scores on the 34 it never saw, with
per-dataset columns differing in both directions by more than the means do (dataset 5:
0.25 train vs 0.43 test; dataset 7: 0.74 vs 0.66 — cells of 2 to 6 neurons, i.e. noise
about which neurons landed where). There is no train–test gap here at all, which is what
11 epochs over 6993 windows looks like; the capacity check shows what actual memorization
would cost (0.994 on 32 windows).

Wall clock, measured stage by stage: data 2 s from the cache (once: a 71 MB download plus
20 csv), smoke test 6.5 s, stage 1 17.4 min, stage 2 44.2 min, stage 3 25.2 min, training
88 s, the single evaluation 6 s — **about 1 h 30 m**, of which the sweeps are 87 min.

## The split discipline

* The split is the notebook's own, rebuilt line for line (`np.random.default_rng(0)`,
  `rng.permutation` per dataset, `n_test = max(1, round(0.2n))`,
  `n_val = max(1, round(0.1n))`): **122 / 18 / 34** neurons, 6993 / 1011 / 1891 windows at
  the shipped window, minus the duplicate → 121 training neurons. No test neuron changes.
* Every script states which splits it reads. **`evaluate_task3.py` is the only file in
  this folder that reads `split["test"]`** and it read it exactly once, after the model was
  trained and every setting was fixed; `baseline.py` can score it but only when passed
  `--test`, so the claim holds of an ordinary pass through the folder.
* The sweep cannot score a test window even by accident: it asks for
  `sf_data.sets(data, ("train", "valid"))`, and `start()` asserts none was materialized.
* Before the single read, the whole evaluation path was run on the validation split with
  `--valid` (`results/eval_valid_log.txt`, `figures/*_valid.png`). Plotting and printing
  code that has never executed is exactly where a final evaluation fails, and a crash
  halfway through would have spent the test split for nothing. That dry run is also what
  caught two stale `bins=` calls and a legend collision, both long before they could touch
  a reported number.
* Nothing was tuned per dataset, and the baseline's σ, the target scales and every
  hyperparameter came from `split["valid"]` alone.

## How to check this yourself

1. `python smoke_test.py` — **6.5 s**, nine `[ok]` lines, asserts on failure. Nothing has
   to be trained first.
2. `python spec.py` — the specification table plus the AS BUILT block, read from
   `results/choices.json`.
3. `python sf_data.py` — re-reads the 20 csv and prints what the audit found: 174 neurons,
   95.6% empty 40 ms bins, the 55–720 s length range, the target scale spanning a factor
   of 265 between the quietest and loudest neuron.
4. `python baseline.py` — the σ grid on the 18 validation neurons, then both baselines on
   train and valid; the test split is opt-in (`--test`) so the one-read claim survives.
5. `python evaluate_task3.py --valid` — the identical evaluation path on validation, about
   4 s, reproducing the trainer's 0.635 from a separate process. This is the check that the
   reported pipeline and the reported number belong to each other.
6. Open `figures/example_neuron.png` and check that the blue transients land on the orange
   spikes. A correlation can hide a systematic error; this picture cannot hide much.
7. `results/choices.json` keeps every arm that lost, with its per-seed scores, its epoch
   curve and its measured seconds — the losers are in the record, not only in this file.

## Honest limits

* **34 test neurons.** The gap over the baseline is 7.2 paired bootstrap sd, so the
  *ranking* is firm; the *level* carries ±0.05 and should be read as 0.59 [0.54, 0.65].
  Differences below about 0.02 on validation — most of stage 3's span — are not results,
  and this report calls them nothing else.
* **Best-epoch selection needs validation labels.** At test time nobody has them. A
  deployment that fixed 24 epochs and kept the *last* one would score about 0.61 on
  validation instead of 0.635, so ≈0.02 of the reported figure is a privilege that only
  exists because the validation labels were available during development. The honest
  number for "this configuration, no peeking" is ~0.61 on validation and correspondingly
  lower on test.
* **The metric cannot see scale.** 0.595 is a claim about the *shape* of the rate, not
  about firing rates: predictions are in units of each neuron's own mean rate and one
  neuron's rate sits at the numerical floor. Anything that needed the absolute rate — a
  Poisson spike-generator, an inference about firing — is out of scope of this number.
* **Per-dataset means rest on 1 to 7 neurons.** Dataset 4's entire row is one cell. The
  dataset-3 loss (−0.013 on 3 neurons) is inside that noise, and the "9 of 10" is stated
  with the exception named rather than smoothed over.
* **Ten datasets are ten labs and indicators, and none is held out as an indicator.** The
  model is one network for all of them and its per-dataset spread (0.42–0.75) is wider
  than its margin over the baseline, so cross-lab generalization is sampled here, not
  measured.
* **Nothing beat the 90-epoch collapse, including the choices.** The curve falls from
  0.635 to 0.52 with the loss still improving; a longer budget, a scheduler or weight decay
  would each be a different experiment, and none was run. The shipped answer is 11 epochs,
  which is early enough to be uncomfortable if the collapse were an artifact of the seed —
  the sweep's 3-seed arms are the only evidence that it is not, and they end at 16–24.
* **Not compared against the challenge's published solutions.** Those are scored on a
  different, held-out test set; comparing a number from here with one from the paper would
  be a comparison of datasets, and the exercise's own two baselines, re-measured in this
  code, are the honest bar.
