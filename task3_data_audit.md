# Task 3 (spikefinder calcium → spikes) — audit of the data cell

What the data actually is, measured on all 174 neurons of all 10 datasets (the
spikefinder training release, exactly what the notebook's data cell loads), and
what to do about it. Nothing here is a modelling choice yet; the choices are at
the end, where they feed the specification table in cell 59.

Figures: `task3_audit_figs/fig1_two_neurons.png` … `fig8_duplicate_neuron.png`.
Raw numbers: `_analysis/sf_inventory.txt` (sections 1-7), `_analysis/sf_audit2.txt`
(8-14), `_analysis/sf_audit3.txt` (15-22), `_analysis/sf_twin.txt`,
`_analysis/sf_twin2.txt`, `_analysis/sf_check_claims.txt`; scripts
`sf_inventory.py`, `sf_audit2.py`, `sf_audit3.py`, `sf_twin.py`, `sf_twin2.py`,
`sf_check_claims.py`, `sf_figs.py` (each takes the data directory as an
argument and is run with `python -I`).

Two of the numbers below are from the **test** neurons: the two baselines of the
task text are quoted against the test split, so they were recomputed there to
check that this is the same task (`0.21` and `0.45` in the text, `0.206` and
`0.467` here). Everything else used to make a decision was measured on the
training and validation neurons.

---

## 1. One neuron is in the data twice, and the split puts the two copies on opposite sides

Dataset 9 has columns 3 and 4. They are the same recording written down twice:

| | column 3 | column 4 |
|---|---|---|
| values as written | `-0.282706008876` | `-0.282701417684` |
| largest difference over 320 s | | **1.24e-05** |
| correlation | | **1.000000** |
| std, length | 1.024, 320 s | 1.024, 320 s |
| spike column | 75 spikes | 75 spikes, **100.0% identical sample by sample** |
| the notebook's split puts it in | **train** | **test** |

![the duplicate](task3_audit_figs/fig8_duplicate_neuron.png)

This is the only true duplicate: of the 1993 pairs of neurons long enough to
compare, one pair sits at 1.000 and the next is at 0.58 (two neighbouring neurons
of dataset 7, whose *spikes* do not correlate at all, so those are two real
neurons sharing a slow trend). Five other pairs are above 0.5 and two of those
also straddle the split, but their agreement lives entirely in the slow band
(2 s-smoothed |r| 0.36-0.78) with spike correlations ≈ 0: shared neuropil or a
shared stimulus, not the same cell.

**What it costs.** `_analysis/sf_twin2.txt` prices the worst case. The derivative
baseline scores 0.443 on the test copy `(9,4)`. If a network memorised the
training copy and scored 1.0 on it, the mean over the 34 test neurons would go
0.467 → 0.484, **+0.016** — three times the noise floor of that mean (problem 2),
and of the order of a fifth of the margin a good network is going to be after here
(the baseline is 0.467 and a deconvolution that works lands somewhere near 0.55).
That is not large, but it is the one structural flaw in the split, and the fix is
free.

### Solutions

**A. Drop `(9,3)` from the training set — recommended.** Its trace differs from
`(9,4)`'s by 1.2e-05 and its spikes are identical, so it carries nothing the test
copy does not already have. Costs 1 of 122 training neurons, changes no test
neuron, keeps every published number comparable, and removes the memorisation
route. One line: `sf_split["train"].remove((9, 3))`.

**B. Report the score twice**, with and without `(9,4)`. Costs nothing, changes
nothing the exercise gave you, but the number you hand in still contains the
leak.

**C. Move `(9,4)` to the training set.** Honest, but the test set becomes 33
neurons and dataset 9 keeps only 3, so you have to recompute both baselines on
the new set before any comparison means anything.

**D. Leave it.** Defensible — a U-Net with per-neuron normalisation is unlikely to
memorise a whole trace — but then do not quote a 0.02 improvement as if it were real.

---

## 2. The metric is noisy: one test neuron is worth ±0.03

94-99% of the 40 ms bins hold no spike (dataset 5: 82.8% zero, dataset 9: 98.5%),
so a neuron's correlation hangs on a handful of bins. Score the *same* baseline
on the first half and the second half of the same recording, and the two
halves disagree by a **median of 0.055** (172 neurons):

![metric noise](task3_audit_figs/fig4_metric_noise.png)

* one test neuron is worth about **±0.03**;
* the mean of the 34 test neurons, about **±0.005**;
* per test neuron the derivative baseline runs from **0.150 to 0.762**, and the
  worse *half* of a recording still scores a median 0.441.

So: differences of a few thousandths between two networks are nothing; the mean
over 34 neurons cannot resolve less than ~0.01; and the per-neuron spread is not
noise you can average away inside one neuron — it is real (some cells are simply
much easier than others).

Also worth knowing before you build anything: **a constant prediction has no
correlation at all.** The metric is undefined for it. `train/valid/test` contain
0 undefined neurons today, but the moment a head collapses (a dead ReLU, an
over-regularised model) a neuron silently drops out — count them and print the
count, do not let `np.mean` hide it.

### Solutions

**A. Fix the protocol before training, not after — recommended.** Decide now that
the model is chosen on the validation neurons and the test set is run once, at
the end; report the mean over the 34 test neurons, the mean over the 10 datasets,
the per-dataset table and the number of undefined correlations. Never tune
against a single dataset's test number: 1-7 test neurons per dataset at ±0.03
each.

**B. Bootstrap the difference you claim.** If you want to say "better than the
derivative", resample the 34 test neurons (say 2000 draws) and report how often
your network wins. With ±0.005 on the mean, a 0.02 gap is ~4 sigma; a 0.005 gap is
nothing.

**C. Do not add more test neurons** by re-splitting. The exercise fixes the split,
and 174 neurons cannot support both a bigger test set and 122 training neurons.

---

## 3. The validation split is 18 neurons, and in five datasets it is one neuron

`round(0.1·n)` on a dataset of 6-13 neurons gives 1. The notebook's split is
**122 / 18 / 34** (train / valid / test), and the validation neurons per dataset
are `1, 2, 1, 1, 1, 1, 4, 2, 2, 3`. Five of the ten datasets contribute exactly
one neuron to the whole tuning signal.

This is not a footnote: it is the reason the exercise's own baseline is
under-specified. Re-tuning the derivative's smoothing width on those 18 neurons
and bootstrapping over them, the width that wins is **80 ms in 41% of the draws,
50 ms in 38%, 30 ms in 11%, 120 ms in 10%** — a coin flip between two answers,
while the whole width curve spans only 0.103 and the mean of 18 neurons is good to
about ±0.022.

### Solutions

**A. Tune few things, and tune them globally — recommended.** One width, one
learning rate, one window: chosen on the mean of the 18 validation neurons. Any
per-dataset or per-neuron tuning on this split is fitting one neuron.

**B. Use leave-one-neuron-out *inside* the training set for the choices that
matter** (window length, depth, when to stop). 122 neurons give ±0.003 per fold
set instead of ±0.022, and the 18 validation neurons stay a clean hold-out for
model selection. Costs compute; the recordings are long, so subsample bins, not
neurons.

**C. Accept the noise and prefer the flat end of a curve.** If 30, 50 and 80 ms
all validate within 0.01, pick the one that is also physically sensible (the
kernel in problem 5 says: the wider one).

---

## 4. Every neuron is on a different scale, and a plain MSE is a different objective from the score

![sparsity and scale](task3_audit_figs/fig3_sparsity_and_scale.png)

| | min | median | max | factor |
|---|---|---|---|---|
| std of the fluorescence | 0.392 | 0.649 | 5.212 | **13** |
| firing rate (spikes/s) | 0.02 | 0.76 | 12.41 | **793** |
| firing rate inside one dataset | | | | up to **203** (dataset 9) |

And the score treats all 174 neurons as equals, while a loss summed over all
samples does not. Target energy (∑ spikes²) is what a plain MSE chases:

* the single largest neuron = **3.9%** of it, the top 10 = **29.6%**, the bottom
  half of all neurons together = **9.4%**;
* per dataset: 17.8% (ds1) … **2.3%** (ds8), where the score gives every dataset
  roughly its share of the neurons.

So an unweighted MSE over concatenated windows spends 30% of its effort on ten of
the 174 neurons and 65% on four of the ten datasets, and is then scored by a mean
over neurons that weights them all the same. The mismatch is worth more than most
hyperparameters.

### Solutions

**A. Normalise the input per neuron, from that neuron's own fluorescence —
recommended.** Subtract its mean, divide by its std, computed from the input trace
only (never from the spikes) — so nothing leaks, and at test time the trace you
normalise with is the trace you are predicting. Without it, BatchNorm has to
absorb a 13× input-scale spread that has nothing to do with spikes.

**B. Make the loss equal-weight neurons too.** Two easy forms, both of which keep
the metric's affine invariance intact:
* *relative rate*: the network predicts the rate in units of that neuron's own
  training-mean rate, so a 20% error on a quiet neuron weighs as much as a 20%
  error on a fast one;
* *per-window mean*: average the loss within a window first, then over the batch
  (this alone stops the fast neurons from dominating a batch).

**C. Do not weight the loss by "importance"** (firing rate, trace length). It
directly contradicts an equally weighted metric.

---

## 5. The calcium kernel is longer than the network's receptive field

Average the fluorescence around every spike (spike-triggered average, 150 neurons
with enough spikes) and you get the kernel the network has to invert:

![kernel length](task3_audit_figs/fig6_kernel_length.png)

| | peak after | back to half | back to 10% | still above a tenth at +5 s |
|---|---|---|---|---|
| fastest dataset (7) | 0.07 s | 0.38 s | 1.04 s | 0 |
| slowest dataset (3) | 0.65 s | 2.33 s | 3.23 s | **5 of 13** |
| all ten | 0.07-0.68 s | **0.38-2.33 s** | 1.04-3.23 s | 11 of 150 |

The notebook's `UNet(dim=1, …, widths=[16, 32, 64])` has a receptive field of
**44 samples = 0.44 s at 100 Hz**, measured by pushing one non-zero sample through
the class taken out of the notebook itself and seeing which output positions move
(`_analysis/sf_check_claims.txt`; every width setting the notebook demos gives the
same number for its depth: 4 levels 92 samples, 5 levels 188). 0.44 s is *below the
half-height* of the kernel in eight of the ten datasets: such a network can see the
rise and nothing else. It cannot do the subtraction a real deconvolution does —
"the fluorescence is still high, so this bump is the tail of an old spike, not a
new one" — and it will systematically over-predict right after a strong transient.

Related: the trace's own autocorrelation halves after 0.12-3.10 s depending on the
dataset, and the calcium lags its spikes by 7-58 samples (0.07-0.58 s).

### Solutions — this is the decision that matters most here

**A. Decimate by 4 to 25 Hz — recommended, and it is nearly free (problem 6).**
The *same* architecture then sees **1.76 s instead of 0.44 s**, with *less* compute
than the 100 Hz version, and one output sample equals one 40 ms metric bin. That
covers the half-height of the kernel in every dataset except 3 (2.33 s) and just
misses dataset 5 (1.79 s); four levels at 25 Hz reach 3.68 s and cover both.

**B. Add levels instead, at 100 Hz:** `[16, 32, 64, 128]` → 0.92 s, `[16, 32, 64,
128, 256]` → 1.88 s. Costs memory (and the deepest level of a 5-level net on a
10 s window leaves 32 samples), and you need five levels to reach what decimation
gives you for free at three.

**C. Dilate.** `dilation=(1,2,4,8)` inside each block multiplies the field without
pooling. Cheap, but BatchNorm + dilation + transposed-convolution upsampling is
more ways to get a subtle bug at the edges.

**D. Give up on the tail and predict the local rise.** This is what the derivative
baseline does, and it scores 0.47. It works — but it is the ceiling you are asked
to beat, so treat it as the floor, not the plan.

---

## 6. The 100 Hz grid is mostly empty space

![spectrum](task3_audit_figs/fig7_spectrum.png)

Of the fluorescence's power, **86.2-100.0% lies below 5 Hz**; the band where the
spectrum has fallen to half its 0.5-2 Hz level is at **1.5-2.8 Hz** in every one of
the ten datasets; the share between 25 and 50 Hz is between 8e-15 and 6.6e-3. The
control that makes this number meaningful: the *same method* applied to the raw
sparse spike trains finds **44.6%** of their power above 25 Hz. So the method sees
high frequencies when they are there — the calcium simply has none.

And the 100 Hz grid is not an interpolation someone did for you, which you would
then have to undo: no zero-order-hold fingerprints (0.000 equal adjacent pairs,
longest flat run 1 sample), no linear-interpolation fingerprints (second
differences exactly zero: ≤ 3e-5 of samples; no repeated first differences; the
smallest gap between two distinct values ~1e-9). It is a genuine 100 Hz acquisition.

Consequence for the metric: the four 10 ms samples inside one 40 ms bin are **not
four independent measurements**, they are ~1. And the same binning is where a lot
of the baseline's score comes from — the very same derivative predictor scores
0.277 at 10 ms bins, 0.460 at 40 ms, 0.687 at 200 ms (problem 7).

### Solutions

**A. Average 4 samples into 1 and work at 25 Hz — recommended.** Nothing above
25 Hz to lose (measured), a quarter of the compute, 4× the receptive field at the
same depth, and the output lands exactly on the metric's grid. Filter *then*
subsample (a 4-sample mean is already enough given where the band ends), not
just "take every fourth".

**B. Keep 100 Hz and let the first layer stride it away:** `Conv1d(1, 16, 8, stride=4)`.
Same effect, learned front-end, a little more compute. Fine if you would rather
the input row of the table say "100 Hz as given".

**C. Keep 100 Hz all the way.** Only worth it if you also solve problem 5 at that
rate. The last chance to use the extra samples is the sub-bin timing of a spike,
and at 0.07-0.58 s rise time with ±ms noise that is not what the 40 ms metric
rewards.

---

## 7. The target is almost all zeros, and the metric's bin size is itself a knob

![baselines](task3_audit_figs/fig5_baselines.png)

At 10 ms, 94.4-99.6% of the samples hold no spike; at the metric's 40 ms,
**82.8-98.5%** of the bins are zero. No neuron is silent and the shortest recording
is 55 s, so nothing is unusable — but a squared error on that target is 95% about
predicting zero.

And the bin size the challenge fixes at 40 ms changes what "good" means
(training neurons, same predictors):

| bin | fluorescence | positive derivative |
|---|---|---|
| 10 ms | 0.111 | 0.277 |
| 40 ms (fixed) | **0.190** | **0.460** |
| 200 ms | 0.338 | 0.687 |

Nobody can game that — the evaluation is fixed — but it explains why the
correlations look modest and why a predictor that is *smooth* at 10 ms is not
penalised: anything constant inside a bin is invisible to the score.

### Solutions

**A. Predict at the metric's resolution — recommended.** One output per 40 ms bin
(25 Hz). It removes 3/4 of the output channels-to-learn, matches the loss to
what is scored, and makes "constant within a bin" a property of the
representation rather than an accident.

**B. Poisson loss on the counts instead of MSE.** The target is a small
non-negative integer with 83-98% zeros; Poisson's gradient is `(λ − y)`, so the
95% empty bins pull weakly and the informative bins are not swamped, and a
`softplus` head keeps the rate non-negative without ReLU's dead zone.

**C. If you keep MSE, weight or normalise it per neuron** (problem 4B) — and
expect to spend capacity on zeros.

**D. Do not optimise the correlation directly** (a `1 − r` batch loss). It is the
metric exactly, but `r` is a statistic over a whole neuron, batches are windows,
and the validation loss in Part 1's `train()` would no longer mean anything.

---

## 8. The tables are ragged: 9% NaN padding, windows, and what must not be scored

![padding](task3_audit_figs/fig2_padding.png)

Recordings are 55-720 s long (median 240 s) and a column is padded with NaN to the
height of its dataset's table: **80 of the 174 columns have some padding, 5,151,326
real samples sit in 5,657,774 cells (9.0% NaN)**, 14.3 h of recording. The good
news, measured column by column: NaN is strictly trailing in all 174 columns, there
are **no interior gaps** (longest dead run = 1 sample) and no dead channels, so the
padding is a pure right edge — and the shortest recording is still 55 s, so every
neuron can give at least two 20 s windows without touching it.

The traps it sets:

1. **NaN poisons everything.** A single NaN in a window makes its loss, its
   correlation and every BatchNorm statistic in the batch NaN.
2. **Zero-filling the tail is not neutral.** It invents a long silent stretch at
   the end of 80 of the 174 neurons — a stretch that looks like "no spikes while
   the fluorescence sits at 0", which is exactly the mapping you do not want to
   learn.
3. **Windows that cross the edge** are partly fabricated, and windows entirely
   inside the padding are scored as if they were data.
4. The metric's own binning drops the last `len % 4` samples; if you bin the
   prediction and the target with different offsets, the correlation is between
   shifted signals.

### Solutions

**A. Cut windows at the end of the real part; never let a window contain padding —
recommended.** Build the window list per neuron from `np.isfinite(trace).sum()`;
pad *inside* a window only at the left edge (reflect or repeat the first sample),
where it is a real signal boundary. Bin the target and the prediction from the
same index 0 of the same real part, so the offsets match by construction.

**B. Mask instead of trimming**: give padded samples target `-1` and skip them in
the loss (as Example 7 does) — needed only if you insist on fixed-length whole
recordings.

**C. Fill the tail with the last value rather than 0** if you must fill: it at
least does not invent a hyperpolarised neuron. Still mask it.

---

## 9. The derivative baseline is not better everywhere, so "beat it in most datasets" has a catch

Test neurons, both baselines rebuilt on this exact split (width chosen on the
validation neurons only):

| | fluorescence | derivative | task text |
|---|---|---|---|
| mean over the 34 test neurons | 0.206 | **0.467** | 0.21, 0.45 |
| mean over the 10 datasets | 0.222 | 0.445 | |

Per dataset (test neurons, fluorescence → derivative): ds1 0.191→0.459, ds2
0.197→0.307, ds3 0.215→0.477, **ds4 0.281→0.259**, **ds5 0.328→0.332**, ds6
0.315→0.578, ds7 0.183→0.476, ds8 0.104→0.554, ds9 0.109→0.369, ds10 0.297→0.641.

**In dataset 4 the derivative is worse than the raw fluorescence** (0.259 vs
0.281) and in dataset 5 the two are level. Both are datasets where problem 5 says
differentiation is the wrong operator: dataset 4 has the fastest kernel of the ten
(0.46 s to half-height, autocorrelation halving in 0.12 s), so its transients are
already over inside a bin and the derivative mostly amplifies noise; dataset 5 has
the least sparse target (17.2% of its 40 ms bins hold a spike) with a slow return
(1.79 s to half), where "how high is it" and "how fast is it rising" carry the same
information. It means two things for you: the bar "beat the derivative in most of
the ten" is really "in 8, and in dataset 4 beat 0.28, not 0.26"; and a network that
*learns* how much to differentiate, instead of always differentiating, has a real
edge to find here.

Also measured, so you do not go looking for a shortcut: predicting a neuron from
the **average of the other neurons of its own recording** scores 0.026-0.027 on
every split. There is no session-level signal to steal.

---

## Checked and clean (so you do not spend time on these)

* **No session leakage.** Neurons of one recording that share a length correlate
  at median |r| 0.063-0.250 after 2 s smoothing — and neurons that do *not* share
  a length correlate at 0.038-0.258. Within-group is no higher than across-group,
  so equal length is not evidence of a shared session, and the per-dataset split
  by neuron is as clean as it looks (apart from problem 1).
* **No interpolation artefacts, no dead channels, no interior NaN, no exactly
  duplicated columns** (by hash), no silent neurons, no neuron too short to score.
* **The tables are honest about their headers**: the first line is `0,1,2,…`, so
  pandas drops no data row; calcium and spikes have identical shapes and
  identical NaN masks; all 174 neurons are accounted for.
* **The spikes are integers 0-6 per 10 ms** (the README calls them "spike rates";
  they are counts), max 6 in a 10 ms bin = 600 Hz, so no saturation problem at
  the target's own resolution.
* **The input is only mildly heavy-tailed**: steps beyond 6 robust σ occur
  4.8-46.7 times per minute, beyond 15σ 0-1.0 times; the largest step of a neuron
  is 7-18× its typical one. Worth a robust or normalised loss, not worth a
  spike-removal front end.
* **The two baselines in the task text reproduce** (0.21 → 0.206, 0.45 → 0.467),
  so the cell loads what the exercise was calibrated on.

---

## The specification table (cell 59): the options, and what I filled in

Each row had a real choice behind it. What follows is the short version of the
menu; the chosen answer is the one already written into cell 59.

**Input.** (i) *25 Hz, 512 samples = 20.5 s, z-scored per neuron* — chosen: no
signal above 25 Hz (problem 6), 4× the receptive field for free (problem 5), one
sample per metric bin. (ii) 100 Hz, 2048 samples, same physical window: 4× the
compute for nothing. (iii) 100 Hz with a strided first layer: same idea as (i),
kept closer to "as given", costs the explicit alignment with the metric.

**Output.** (i) *one rate per 40 ms bin, same length as the input, every position
used* — chosen. (ii) one rate per 10 ms sample at 100 Hz: faithful to the task
text's "sample by sample", but 4/5 of its resolution is invisible to the score.
(iii) a spike probability per sample + a smoothing step: a different task
(classification) than the regression the metric scores.

**Last layer.** (i) *1×1 conv + softplus* — chosen: a rate is non-negative and
unbounded above. (ii) ReLU: fine, with a dead zone exactly where 95% of the bins
live. (iii) linear: lets the network predict negative rates, which the metric
punishes for no information. (iv) sigmoid: bounded, wrong.

**Loss.** (i) *Poisson NLL on the 40 ms counts, rate expressed relative to each
neuron's own training-mean rate, averaged within a window then over the batch* —
chosen: it fits a sparse count target (problem 7) and it weights neurons equally,
like the score (problem 4). (ii) per-neuron-normalised MSE: simplest thing that
fixes both, use it if Poisson trains unstably. (iii) plain MSE: the mismatch in
problem 4, 30% of the weight on 10 neurons. (iv) a direct `1 − r` batch loss
(problem 7D).

**Metric.** Only one honest answer — the challenge's own. What is a choice is
*how you report it*: mean over the 34 test neurons **plus** the per-dataset table
**plus** the 10-dataset mean (they differ: 0.467 vs 0.445), the undefined count,
and the ±0.005 noise floor (problem 2) stated next to the number.

**Baseline.** Both, rebuilt here rather than quoted: 0.206 and 0.467, with the
ds4 exception (problem 9). The number to beat is 0.467 on the mean and 0.281 in
dataset 4.

### Things the table cannot hold but the implementation must

1. **Model selection.** Part 1's `train()` early-stops on `loss_fn`, and here the
   loss (Poisson) is not the metric (Pearson). Report the validation *correlation*
   alongside the loss and pick on it; otherwise you keep the epoch that best fits
   the zeros.
2. **No per-neuron tuning** — 5 datasets have one validation neuron (problem 3).
3. **One line for the duplicate**: drop `(9,3)` from the training set (problem 1A).
4. **Determinism**: windows are generated from `np.isfinite(...).sum()` per neuron,
   not from the padded table height, so a change in window length cannot silently
   pull padding into a batch (problem 8).
