# Task 6 (colourisation of photos) — audit of the data cell

What the data actually is, measured on all 13,394 photos of Imagenette2-160 exactly as
the notebook's data cell loads them (`sorted(glob)`, short side → 64 with BICUBIC, central
64×64, seed-0 permutation of the val folder), and what to do about it. Nothing here is a
network: every ceiling quoted below was produced by a lookup table or by the truth itself,
so the "which route is worth training" question is answerable before one epoch runs. The
modelling choices are at the end, where they feed the specification table in cell 77.

Figures: `task6_audit_figs/1_the_alpha_dial.png` … `6_the_photos.png`.
Raw numbers: `_analysis/col_inventory_log.txt` (sections 1-7), `_analysis/col_oracle_log.txt`
(sections A-H), `_analysis/col_dups_log.txt` (sections 1-5); scripts
`_analysis/col_inventory.py`, `_analysis/col_oracle.py`, `_analysis/col_dups.py`,
`_analysis/col_figs.py` — each takes the data directory as an argument and is run with
`python -I`. The arrays are cached in `_analysis/col_cache/` and the reload was checked
against the files: re-running `load_photo` on 400 photos reproduces the cached bytes
exactly, so every number here is computed on the same pixels the notebook will train on.

Two conventions of "the PSNR of a set of photos" are in use in this exercise and they rank
the predictors **oppositely**, so both are printed everywhere: the task's "about 25 dB" for
gray is the per-photo PSNR averaged over photos, while the notebook's only PSNR code
(Task 4, cell 30 — `10*np.log10(1/F.mse_loss(a, b))` over a whole batch) is the pooled one.

Three numbers below are from the **test** photos, and only for the reason the task text
quotes test numbers: the gray baseline (25.36 dB against "about 25") and the photos'
colourfulness (0.041 against "about 0.04"), plus the definition search that identifies which
colourfulness that is. No fitted predictor was ever scored on the test split. Everything
used to make a decision was measured on train (fit) and valid (decided).

---

## 1. The task's own first metric is maximised by refusing to colour

This is the trap the task is built around, and it is deeper than "L1 gives dull colours".
Take the L2-optimal per-pixel answer — the conditional mean of the two chroma channels given
luminance, fitted on all 9469 training photos, which is the best answer any per-pixel
regressor can converge to — and turn the saturation down with a dial α
(predicted chroma = α · E[chroma | Y]). α = 1 is that optimal answer, α = 0 is the gray
baseline. Then measure. On the 100 validation photos:

| α | PSNR, per photo then mean | PSNR, pooled | chroma L1 | hue error | colourfulness |
|---|---|---|---|---|---|
| 0.00 (= gray) | **26.94** | 21.06 | 0.0804 | 96.6° | 0.000 |
| 0.40 | 24.76 | 21.24 | 0.0792 | 60.0° | 0.004 |
| 1.00 (= L2-optimal) | 24.36 | 21.39 | 0.0794 | 60.0° | 0.011 |
| 1.30 | 24.04 | **21.41** | 0.0807 | 60.0° | 0.015 |
| 2.00 | 23.48 | 21.31 | 0.0843 | 60.0° | 0.022 |
| truth | ∞ | ∞ | 0.0000 | 0.0° | 0.040 |

The per-photo-then-mean PSNR **peaks at α = 0** and falls monotonically: on that convention
the best possible photo is no colour at all. The pooled PSNR peaks at α = 1.30, where theory
says it must (it is the minimiser of the pooled MSE). Colourfulness rises monotonically with
α and never reaches the photos' 0.040 without going far past both optimums. Hue error is flat
at 60° for every α from 0.2 to 1 — it separates none of them.

And the same holds for the answer the bin route produces, and for the truth's own best:

| predictor (all fitted on train, scored on the 100 valid photos) | PSNR per photo | PSNR pooled | chroma L1 | hue | colourfulness | mean \|chroma\| |
|---|---|---|---|---|---|---|
| gray baseline | **26.94** | 21.06 | 0.0804 | 96.6° | 0.000 | 0.0 |
| conditional mean (route A's ceiling, L2-optimal) | 24.36 | 21.39 | 0.0794 | 60.0° | 0.011 | 4.5 |
| L1-optimal chroma cell per brightness | 24.02 | 21.27 | 0.0832 | 63.3° | 0.015 | 5.6 |
| argmax bin, γ = 0 (route B's ceiling) | 23.99 | 21.18 | 0.0836 | 68.8° | 0.015 | 5.6 |
| argmax bin, γ = 0.5 (route B+) | 19.58 | 18.57 | 0.0968 | 75.0° | 0.022 | 8.8 |
| argmax bin, γ = 1.0 | 7.87 | 7.78 | 0.4820 | 93.7° | 0.236 | 95.9 |
| conditional mean + 3×3 neighbourhood context | 24.29 | 21.42 | 0.0794 | 58.9° | 0.012 | 4.7 |
| a predictor *told* the true class, painting its mean chroma | 24.35 | — | — | — | — | — |

The paired numbers, which is the honest way to compare on 100 photos (the photo-to-photo
spread cancels): the conditional mean is **−2.581 dB** against gray (sd 1.442, 95% CI
[−5.715, −0.063]), and gray wins outright on 39 of the 100 photos. On the pooled-MSE basis the
same comparison is **+0.334 dB in favour of the mean** (95% CI [+0.141, +0.525]) — the theory
is not violated, only the convention is unhelpful.

**Why the mean over photos does this.** PSNR is a logarithm of a per-photo MSE, so a photo the
baseline reproduces to within a rounding error is worth an enormous number of decibels, and a
photo 65 chroma units wide is worth very few whichever way you go:

| colour quintile of the valid photos (mean chroma radius) | gray dB | mean dB | Δ | gray wins |
|---|---|---|---|---|
| 1 (0-8) | **46.32** | 32.07 | −14.25 | 85% |
| 2 (8-11) | 26.56 | 26.54 | −0.02 | 35% |
| 3 (11-16) | 24.22 | 24.44 | +0.22 | 35% |
| 4 (16-23) | 20.76 | 21.47 | +0.70 | 15% |
| 5 (23-65) | 16.86 | 17.29 | +0.43 | 25% |

Correlation between a photo's colour content and its gray PSNR: **−0.51**. On the colourful
photos — the only ones where colourising is a real achievement — the conditional mean beats
gray on 65-85% of them. The average over photos throws that away.

### Solutions

- **A. Report the metric triple, never the PSNR alone, and state the convention.** PSNR
  (per photo, clipped, see problem 2), chroma L1 and colourfulness, plus the count of photos
  where the network beats gray. This costs nothing and it is what the task's own "what to
  show" asks for. Recommended, and it is already in the table.
- **B. Add a clipped-mean and a median PSNR.** The median of the gray baseline is 23.95 dB on
  valid against a mean of 26.94, and on the colourful half the network's advantage is what
  anybody looking at the photos would call the result. Recommended next to A.
- **C. Score PSNR on chroma only** (the two channels, with Y written back exactly). It removes
  the 99% of the pixels that carry no information about the model's quality and it makes
  gray's own score a real number rather than a large one. Not the metric the task quotes, so
  as an extra column, not a replacement.
- **D. Switch to the pooled convention.** It agrees with the theory, but it is a *worse*
  guide here: the difference it rewards is +0.33 dB against a 25 dB baseline while the
  visible difference between α = 0 and α = 1 is "grey versus tinted". Do not adopt it
  silently just because cell 30's code happens to compute it.
- **E. Optimise PSNR as the training objective.** The sure way to lose this task: the
  argmax-of-PSNR answer is the gray image.

---

## 2. "About 25 dB" is at least four different numbers, and the difference is bigger than most results

Counted on exact RGB bytes: **3 of the 100 validation photos and 8 of the 500 test photos
have R == G == B in every one of their 4096 pixels** — source photographs that were black and
white before anyone resized them. The gray baseline reproduces those perfectly, so their MSE
is 0 and their PSNR is whatever the epsilon in the code says it is. The same baseline is:

| convention | valid | test |
|---|---|---|
| mean of per-photo PSNR, MSE floored at 1e-12 (what the code does) | 26.94 | 25.36 |
| each photo clipped at 60 dB | 25.14 | 24.40 |
| each photo clipped at 40 dB | 24.46 | 24.06 |
| the exactly-grey photos dropped | 24.06 | 23.82 |
| **median** per-photo PSNR | 23.95 | 23.21 |
| pooled MSE | 21.06 | 21.11 |

A spread of 3 dB on the *baseline itself*, none of it about the model. The task's "about 25 dB"
is reproduced by the plain floor (25.36 on test), so the notebook and the text agree — but a
claim of the form "+0.4 dB over the baseline" is meaningless until the convention is written
down, and the max per-photo PSNR in the arrays is 120.0 dB, which is `10*log10(1/1e-12)` and
not a property of any photograph.

### Solutions

- **A. Clip each photo's PSNR at 40 dB before averaging, and report the median next to the
  mean.** 40 dB is the usual choice, it costs nothing on ordinary photos (only the exact
  matches and near-perfect ones are affected) and it makes the baseline a stable 24.46/24.06.
  Recommended; it is in the table.
- **B. Leave the 1e-12 floor and state it.** Legitimate, and it is what makes the task's 25 dB
  reproducible, so keep the number *as the baseline* — but then never quote a difference
  without the same floor.
- **C. Drop the exactly-grey photos from the metric and keep them in the figure grid.** They
  are honest data, and showing them is interesting (a colouriser must learn to leave them
  alone) — but silently dropping 1.6% of the test set changes the denominator of every claim.
  Only do this if the number is labelled "492 photos".
- **D. Never write `10*log10(1/mse)` without the `maximum(mse, eps)`** — a zero-MSE photo is
  also a `-inf` → `nan` risk the moment a model is evaluated on a photo it reproduces exactly.

---

## 3. Neither "colourfulness" nor "the chroma L1" is defined anywhere in the notebook

Cell 78 quotes "a colourfulness of 0 (the true photos: about 0.04)" and cell 78 asks for "the
PSNR, the chroma L1 and the colourfulness". No cell defines either function, and `skimage` —
named in cell 75 for `rgb2lab` — is not installed here at all. Colourfulness is not one
function; it is at least six, and they differ by 3.4×:

| definition | train | valid | test | gray | \|test − 0.04\| |
|---|---|---|---|---|---|
| Hasler–Süsstrunk (std + mean of rg, yb) | 0.112 | 0.139 | 0.138 | 0.0000 | 0.098 |
| same, mean term only | 0.095 | 0.134 | 0.133 | 0.0000 | 0.093 |
| pooled std of (Cb, Cr) | 0.063 | 0.089 | 0.089 | 0.0000 | 0.049 |
| mean chroma **radius** | 0.065 | 0.061 | 0.062 | 0.0000 | 0.022 |
| **mean \|chroma\| over pixels and both channels** | 0.042 | 0.040 | **0.041** | 0.0000 | **0.001** |
| RMS chroma | 0.065 | 0.061 | 0.062 | 0.0000 | 0.022 |

Every candidate scores the gray baseline at exactly 0 — a gray image has no chroma and no
channel spread by construction — so the task's "0 for gray" cannot identify the definition.
Only the second number can, and mean absolute chroma is the only one that lands on 0.04
(1.5× off for the radius, 2.2× for the pooled std, 3.4× for Hasler–Süsstrunk). So this audit
uses — writing Cb, Cr centred, i.e. neutral = 0, which is what the code in `_analysis/` does
(JPEG's 0-255 Cb is this plus 128, so \|Cb−128\| in that convention is the same number):
**colourfulness = mean over pixels *and* over both channels of \|Cb\| and \|Cr\|, divided by
255**. And **chroma L1 = mean over pixels of (\|ΔCb\| + \|ΔCr\|) / 255**, which for gray comes
out at 0.0804 — exactly twice the photos' colourfulness, as it must when the prediction is
neutral.

### Solutions

- **A. Write the three functions yourself, in one cell, with the formula in a comment, and use
  those numbers everywhere** (recommended; the definitions above are what the table in cell 77
  now names). Nothing needs `skimage`.
- **B. Use Hasler–Süsstrunk** because it is the citable one — but then the real photos score
  0.138, not the 0.04 the task quotes, and the reader cannot check your number against the
  text. If you want it, report it as a fourth column with its own label.
- **C. Ask.** Cheap to ask the tutor, cheap to answer, and it removes a whole class of "my
  number does not match the handout". Do A regardless; C in passing.

---

## 4. The luminance carries 1% of the colour, so both routes have low ceilings — and they are measurable now

Conditioning on Y with a 64-bin lookup on 9.7M sampled training pixels:

- the residual chroma variance is **99.0%** of the total: Y explains 1.0% of the colour;
- the Y-conditional mean has \|mean\| = 3.93 chroma units at the median bin against a
  within-bin spread of 16.6 — ratio 0.25 at the median, 0.41 at its best;
- the same thing per brightness band is figure `3`: the answer's size is a quarter of the
  question's size everywhere between Y = 40 and Y = 220;
- the class label is worth a little more but still little: **8.06%** of the pixel-wise chroma
  variance (class mean chroma radius 1.7 to 12.3 on a spread of 23.6), and a predictor *handed
  the true class* and painting its mean chroma scores 24.35 dB against gray's 26.94.
  Recognition buys colour that much and no more.

What that implies for the two routes, measured rather than argued:

- **route A's ceiling is the conditional mean**: 24.36 dB per photo / 21.39 pooled / colourfulness
  0.011. A wider network, more epochs, a U-Net, context, a pretrained encoder — none of it
  can beat the conditional mean *on a per-pixel L2 objective with only Y as input*, because the
  conditional mean **is** the optimum of that objective. Adding a 3×3 neighbourhood of the
  luminance moves the pooled PSNR by +0.03 dB (21.42 vs 21.39) and the hue error from 60.0° to
  58.9°: local context is worth almost nothing when the ambiguity is object-level.
- **route B's ceiling at a given number of bins** is the PSNR of the *binned truth* (which sees
  the answer, so it is a hard ceiling):

| bins per axis | K | PSNR of the binned truth (valid) | colourfulness |
|---|---|---|---|
| 6 | 36 | **22.31** | 0.084 |
| 12 | 144 | 28.95 | 0.056 |
| 16 | 256 | **31.54** | 0.050 |
| 32 | 1024 | 37.65 | 0.043 |
| 64 | 4096 | 43.85 | 0.041 |

  A 6×6 = 36-class colour vocabulary **cannot** beat the gray baseline (22.31 against 26.94)
  even with a perfect classifier and even though it is looking at the truth. This is the number
  to know before choosing "a few hundred bins".
- YCbCr is the better grid at equal cost: at 8/16/32 bins per axis it gives 25.09/31.54/37.65 dB
  against Lab's 23.36/30.08/36.29; and at 16 per axis only **89 of the 256 cells are populated
  at all**, the largest holding 24.3% of the pixels. (Hand-rolled Lab, D65, round-trip
  ≤ 0.005 of 255 — `skimage` is not installed, so it had to be written anyway.)
- The information-theoretic floor for route B, at a 32×32 grid: H(C | Y) = 3.343 nats
  (perplexity 28 cells) against H(C) = 3.563 nats (35). All the luminance tells you about
  colour is worth **0.220 nats**, and the best possible per-pixel top-1 accuracy is
  **18.8%**. If you report "accuracy" on the bins, that is the number to be compared with, and
  it explains why a good-looking photo and a terrible-looking accuracy go together. Both figures
  are the 32×32 chroma grid; at the 16×16 grid that solution A recommends, the same table on the
  same Y conditioning gives H(C | Y) = **2.377 nats** against H(C) = 2.550 (so luminance is worth
  0.173 nats there) and top-1 **30.12%** — the grid has to be
  stated with the number, and `task6_colorize/metrics.py::ceilings` prints the pair for any grid.
  The cells are populated differently in the two samples (89 of 256 on the 100 validation photos
  above, 193 of 256 on the 9469 training photos, top cell 24.3% and 26.66% respectively): that is
  sample size, not a disagreement, and it is why the trained model's own ceiling is measured on
  the training split.

### Solutions

- **A. Choose K ≥ 144, preferably 256 (16×16 on Cb–Cr).** Below that the ceiling is below the
  baseline; at 256 the ceiling is 31.5 dB and 89 of the 1024 cells at 32×32-scale carry 24% of
  the pixels. Recommended, and it is in the table.
- **B. Give route A a calibration dial instead of a bigger network**: train the L1 regressor,
  then pick one scalar α on valid (a 20-point grid over [0.8, 2.0] costs one pass over 100
  photos). It is the only way route A gets its colourfulness above 0.011, and it is honest —
  you are stating, in one number, how much the L2/L1 answer under-sells the colour.
  Recommended. Note the whole α curve in the report; a single tuned α looks like magic.
- **C. Use the class labels / a pretrained classifier to condition the colour.** Measured
  ceiling: 8% of the chroma variance, and it loses to gray on PSNR. Not worth it for this task.
- **D. Change the input, not the head**: a colouriser that sees more context (larger receptive
  field, deeper encoder) attacks the 99% that is *not* in this pixel's Y — but only if it can
  actually see objects, which a small 64×64 U-Net mostly cannot. Worth saying in the write-up
  as the reason a bigger net changes the *hue* more than the *PSNR*.
- **E. Do not report "top-1 accuracy on the bins" without its ceiling next to it — and read it at
  the grid you trained on.** The 18.8% above is the 32×32 grid; the 16×16 grid this audit
  recommends in solution A allows **30.1%** (30.12%, measured on the 9469 training photos by
  `task6_colorize/metrics.py::ceilings`, whose 32×32 answer is the 18.8% above to two decimals —
  same lookup table, same 64-bin conditioning on Y, finer chroma grid). Two qualifiers belong in
  the same breath as either number: it is what a colour **lookup table** scores, chroma read off
  one 64-bin number per pixel, so a network that sees luminance at full resolution and in a
  neighbourhood is allowed to beat it — that is a result, not a contradiction. The one figure
  that really caps route B is the PSNR of the binned truth, 31.54 dB, which no decoder limited to
  bin centres can pass.

---

## 5. The regression answer is grey by construction — the "dull brownish gray" is now a number

The task text's claim ("the colour that minimizes the average error over all possible answers
is a dull brownish gray") is not a hypothesis here, it is arithmetic on this data:
\|E[chroma | Y]\| is 4.5 of 128 where the photos average 15.7, so the mean answer keeps
**29%** of the colour (25% by the median-bin ratio). The measured colourfulness of the
L2-optimal answer is 0.011 against the photos' 0.040 and the gray baseline's 0.000 — the
regressor gets about a quarter of the way from gray to real. Switching the loss from MSE to L1
moves it to 0.015 (the L1-optimal cell, hue error 63.3°), which is a real but small gain, and
it costs PSNR (24.02 against 24.36).

Also worth knowing before you interpret the density plots: the conditional distribution is
*elongated along a line through the origin* (the neutral axis, plus the sky/soil directions)
rather than multimodal with separated modes, at least at this resolution. The most likely
single colour per brightness and the mean land nearly on top of each other (figure `3`, × and
○ within a couple of units everywhere). So the "a car may be red or blue" story is, on
Imagenette at 64×64, mostly "a pixel may be any shade along one direction, or nothing at all"
— which is why route B's argmax is barely more colourful than route A's mean at γ = 0
(0.015 vs 0.011).

### Solutions

- **A. Train route A with L1** (not MSE), on chroma in units of ±128, and treat its output as a
  *direction* to be scaled: calibrate one α on valid. Recommended.
- **B. Add a colourfulness or an adversarial term to route A's loss.** It is the honest way to
  make route A colourful, but it is a different objective from the one the task names, and a
  GAN is a second network to debug. Optional, marked as such.
- **C. Accept the dullness and report it as the finding**: route A reaches colourfulness 0.011
  at 24.3 dB, the task says it "colours almost nothing", and you have the curve that shows why.
  This is what the task is actually asking for — do it whichever of A or B you also do.
- **D. Predict residual chroma rather than chroma** (i.e. regress Δ from the previous frame or
  from a smoothed version of the same image). There is no second frame here, so no.

---

## 6. Route B's traps, all three, measured

1. **The bin space is mostly empty and the head is dominated by grey.** At the 32×32 grid the
   9469 training photos' 38,785,024 pixels populate **696 of 1024** cells; the top cell holds
   **15.52%** of all pixels (it is the neutral one), **534 cells hold fewer than 100 pixels and
   409 hold fewer than 10**. An unweighted softmax spends most of its capacity on cells that
   barely exist and is rewarded for predicting grey.
2. **The prior pull is a cliff, not a dial.** Zhang et al.'s fix is to pick
   argmax (log p − γ · log p̄). Measured on the same pixels, per bin:

| γ | PSNR per photo | chroma L1 | hue | colourfulness | mean \|chroma\| |
|---|---|---|---|---|---|
| 0.0 | 23.99 | 0.0836 | 68.8° | 0.015 | 5.6 |
| 0.5 | 19.58 | 0.0968 | 75.0° | 0.022 | 8.8 |
| 1.0 | **7.87** | 0.4820 | 93.7° | 0.236 | **95.9** |
| 1.5 | 7.33 | 0.5190 | 94.2° | 0.255 | 102.2 |

   At γ = 1 the predicted chroma magnitude is 95.9 of 128 — every pixel painted at the edge of
   the colour space, PSNR 7.9 dB, i.e. garbage. The published paper works in *ab* after
   normalising by a per-image σ and pulls with γ ≈ 1 *there*; copying γ = 1 onto unnormalised
   CbCr of these photos produces neon. The usable window is 0 to about 0.5.
3. **The metric you can compute for route B is the misleading one.** Top-1 bin accuracy has a
   ceiling — 18.8% at the 32×32 grid it was measured on above, 30.1% at the 16×16 grid solution A
   recommends — and it is a ceiling on a colour *lookup table*, not on a network; the same
   model's photos can be visibly better than route A's. And route B's
   output is piecewise constant in colour, so its PSNR is capped by its own binning (31.5 dB at
   K = 256) no matter how good the classifier is.

### Solutions

- **A. γ ≤ 0.5, chosen on the validation photos, and reported as a tuned constant with its
  curve.** Recommended: the window where B+ is colourful *and* not broken is between about 0.1
  and 0.5. Quote colourfulness alongside so "colourful" is not a vibe.
- **B. Normalise the chroma before binning** (divide Cb, Cr by a per-dataset σ, or bin in the
  "quantised σ-scaled" way the paper describes) so that γ has the same meaning it has in the
  paper. Recommended if you want γ ≈ 1 to be meaningful; it is a slightly bigger change to the
  target, so say so in the table.
- **C. Weight or resample the bins** (class-balanced softmax / frequency prior subtraction at
  *train* time rather than at inference). Directly attacks the 15.5%-grey-cell imbalance.
  Reasonable, and it is what "B+" is often implemented as — but then state which of B/B+ the
  γ-pull and which the reweighting is, because two different "B+" implementations exist.
- **D. Keep K = 256 and drop the cells with < 100 pixels into their nearest populated
  neighbour** (534 cells today). Fewer, better-populated classes; the ceiling loss from
  merging the tail is small because the tail is where almost no pixels are.
- **E. Do not "fix" the empty cells by merging the whole plane to 36 bins.** Measured ceiling
  22.31 dB, below gray. This is the trap the task means by "its own trap".

---

## 7. The validation split is 100 photos with a ±1.7 dB noise floor, while 3325 photos sit unused

The data cell takes `order[500:600]` as valid and `order[:500]` as test, out of the 3925
photos in `val/` — so **3325 of the 3925 val photos are used by nobody**, and every modelling
decision must be made on 100 photos. Bootstrapping over photos:

| | mean | bootstrap sd | 95% CI | per-photo sd |
|---|---|---|---|---|
| gray on valid (100) | 26.94 | **1.746** | [23.93, 30.65] | 17.39 |
| gray on test (500) | 25.36 | 0.597 | [24.24, 26.56] | 13.23 |

So a difference of +0.5 dB in the mean of 100 photos is noise; the paired version is the only
one with power (the per-photo mean−gray difference has sd 1.44 dB, a standard error of 0.14 dB
on 100 photos — enough to see the −2.58 dB of problem 1 clearly, but not enough to rank two
networks that differ by 0.3 dB). And the 100 photos are not even balanced: **valid has [16, 14, 12, 11, 11,
9, 8, 8, 7, 4] photos per class** (the smallest class is 4 photos — one photo's PSNR there can
move that class's average by 4 dB), against [60…41] on test and [993…858] on train.

### Solutions

- **A. Make decisions on the paired statistic over the 100 photos** (per-photo difference, its
  mean and its 95% CI), and say "gray wins on k of 100". Recommended; it is a two-line
  function and it is the only comparison on 100 photos that does not lie.
- **B. Enlarge the validation split from the 3325 unused photos** — e.g. `order[600:1600]`,
  1000 photos, cutting the noise to ~±0.55 dB. **Do it without touching the test split**, and
  say in the write-up that the extra photos come from the part of `val/` the data cell ignores.
  Recommended for anything you actually want to believe; keep the original 100 as the headline
  numbers so they are comparable with the handout.
- **C. Report per-class means with n**, so a 4-photo class cannot masquerade as a result.
- **D. Fix the seed everywhere** (loader seed 0 is fixed; set the training seed too), because
  on 100 photos a different initialisation moves the mean more than most architecture choices.

---

## 8. The loader's own two decisions: the centre crop, and a 2.5× downsample with no pre-filter

`load_photo` resizes the short side to 64 and cuts the centre 64×64. Measured on the sources
(all of them exactly 160 px on the short side, so nothing is ever upscaled):

- the crop keeps a **median 75%** of the resized photo (10th pct 67%, minimum **20%**);
  aspect ratios run to **5.01:1** in train; **4%** of training photos lose more than 40% of
  their area and **2%** lose more than half;
- what is lost is always the edges, never random pixels — a systematic truncation of the
  scene. The subject is usually in the centre of an Imagenette crop, which is why this is
  tolerable, but a "fish" photo that is 80% water at the edges is a different task after the
  crop (figure `6`, top row);
- PIL's `resize` does not pre-filter when shrinking, so BICUBIC 160→64 aliases. Against
  LANCZOS with proper support it disagrees by **1.24 of 255 per pixel** — a mutual PSNR of
  **42.8 dB**. The same error is in the input and in the target, so it is self-consistent and
  it costs no PSNR, but it is a floor under how much fine colour detail (thin saturated edges,
  coloured textures) any model here can ever show, and it is the same order as the 25 dB the
  whole task is arguing about.

### Solutions

- **A. Keep the data cell as it is.** It is the notebook's loader, it is self-consistent, and
  changing the resampling changes every number including the baseline. Recommended.
- **B. Say what the crop does in the write-up**, and put the widest photo with its crop box in
  the figures (done: `6_the_photos.png`) so the reader sees that 20% of a photo can be the
  input.
- **C. If you want to measure the loader's effect rather than inherit it**: load the same photos
  at 128 and crop, or with LANCZOS, train on it, and check that the conclusions are unchanged.
  It is a fine robustness check and it is *not* something to mix into the reported baseline.
- **D. Do not "fix" it with a random crop at train time.** It is legitimate augmentation for
  segmentation-style tasks, but here it changes the input/target pair the task defines, and the
  validation/test photos would still be centre-cropped, so the split mismatch would be yours
  to explain.

---

## 9. A tenth of the pairs carry no colour at all, and they dominate a per-pixel loss

Per photo, mean chroma radius: median 14.3 (train) / 12.6 (valid) / 13.9 (test);
**10.6% of training photos (1003) are below radius 5**, 14.0% of the valid ones and 11.0% of
the test ones, and **181 / 3 / 8 photos are *exactly* R == G == B** in all 4096 pixels.
Per pixel: **49.2% of training pixels are within 10 of neutral and 31.0% within 5**.

Two consequences. For the loss: about a third of the pixels have essentially nothing to learn,
and their optimal answer is 0 — which is exactly the collapse the task warns about, arriving
through the data as well as through the arithmetic of problem 5. For the metric: those photos
are the ones the PSNR average is made of (problem 1's first quintile).

### Solutions

- **A. Keep them. Do not filter.** They are real photographs, and a colouriser that can tell
  "this photo was black and white" from the luminance is doing the right thing. Recommended.
- **B. Report the metric separately for the colourless and the colourful half** (or the
  quintiles of problem 1). This is the single most informative extra column in the whole task,
  and it is free. Recommended.
- **C. Weight the loss by the true chroma magnitude** (or restrict it to \|chroma\| > 5 pixels).
  It fights the collapse and it is one line, but it changes the objective away from the one the
  task names and it will *lower* the PSNR — do it as an experiment labelled as one, or not at
  all. Not recommended as the main run.
- **D. Never report "colourfulness of my predictions 0.011" without the 0.040 of the truth and
  the 0.000 of gray on the same line**, which is precisely what the 30%-grey-pixel content makes
  easy to get wrong.

---

## 10. Chroma is spatially coherent, so a per-pixel metric is mostly a per-region metric

Semivariogram of Cb on the training photos (half the variance of the difference between two
pixels `lag` columns apart, divided by the variance of Cb; 0 = identical, 1 = unrelated):

| lag | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| normalised | 0.03 | 0.08 | 0.15 | 0.26 | 0.38 |

A neighbouring pixel carries **97%** of the information of a free measurement: colour comes in
regions. One wrong region of 100 pixels is 40 wrong "independent" pixels, and 4096/100 = 41
photos' worth of weight if the region is the sky. Practically: the effective sample size of
"per-pixel accuracy" on 4096 pixels is more like a few hundred, and the task's "a grid of 8
test photos" is not decoration — it carries most of the signal about whether the model works
that the 500-photo average smooths away.

### Solutions

- **A. Make the 8-photo grid a first-class deliverable, chosen by a rule** (e.g. the most
  colourful, the least colourful, and six spread over the colour quintiles), not the first
  eight of `test`. Recommended.
- **B. Where you would quote a per-pixel standard error, use the effective-sample-size
  correction** (ρ at lag 1 ≈ 0.97 → n_eff ≪ 4096), or simply quote the per-photo number, which
  already respects the structure.
- **C. Consider a small bilateral/total-variation term at inference** to keep colour regions
  coherent. Cheap and it helps the eye; it changes the PSNR by little. Optional.

---

## Checked and clean (so you do not spend time on these)

- **The split does not leak.** 0 byte-identical photos inside any split and 0 between train and
  valid/test (blake2b over the file bytes). The 8×8 average hash flagged 6 *candidates* in
  test, all refuted by looking. Full nearest-neighbour search: over the 600 valid+test photos
  against all 9469 training photos, the closest pair of the whole search differs by MAD 11.5
  on the 64×64 luminance where two random training photos differ by 76.0; median 53.8, 10th
  pct 37.9; 1 photo of 600 is below MAD 15. The four closest pairs are in
  `task6_audit_figs/4_no_leak.png` — different photographs of similar scenes (same class in
  every case, 26% against 10% by chance, which is what two photos of one ImageNet class look
  like at 64×64). Unlike Task 3, there is nothing here to fix.
- **Decoding.** 13,394 photos, 0 failures; no truncated files, no CMYK, no progressive-JPEG
  surprises.
- **EXIF orientation.** Every photo carries orientation tag 1 (or none), so `load_photo` never
  needing `exif_transpose` costs nothing on this archive. It is still a one-line landmine for
  any other photo set, so write the line or comment why it is not needed.
- **No upsampling.** 0 of 10,069 photos are smaller than 64 px on the short side, so the loader
  never invents pixels.
- **Nothing to collide.** 0 filename collisions inside a split and 0 shared names across the
  train / valid+test line, so a dict keyed by filename would have been safe here too.
- **The archive's one piece of cruft is inert.** `train/.DS_Store` sits at the top level;
  `glob('*/*.JPEG')` matches neither the top level nor a non-JPEG name, so it never reaches
  the arrays. (Checked directly: 9469 + 3925 `.JPEG`, nothing else inside the class dirs.)
- **The class balance in train is fine** (858-993 per class), and all ten classes appear in
  every split.
- **Luminance is sane**: Y from 2 to 255, median per-photo detail (std of Y) 56; 20 training
  photos are near-flat fields, which is the same phenomenon as problem 9, not a defect.
- **`col_rgb` is exactly what the cell says**: (9469, 64, 64, 3), (100, …), (500, …) uint8,
  and the reload check above proves the cached copies are byte-identical to fresh loads.

---

## The specification table (cell 77): the options, and what I filled in

The table is one column of answers but the task's "what to show" asks for it twice — for the
regression (A) and for the bins (B, B+) — so the cell now has two answer columns. The options
considered first, then what went in.

**Network input (shape).** Options: `(B, 1, 64, 64)` luminance in [0,1]; `(B, 3, 64, 64)` the
gray photo as three identical channels; `(B, 3, …)` with the *true* RGB (never — that is the
answer); Y and Y again as two planes (no). *Filled:* `(B, 1, 64, 64)`, float, `Y = rgb @
[0.299, 0.587, 0.114] / 255`, the same weights that make the target — one channel, because
three copies of it are the same information at triple the input weights.

**Network output (shape).** Options for A: `(B, 2, 64, 64)` chroma; `(B, 3, 64, 64)` full RGB
(the network then has to learn to copy Y through one channel — wasted capacity and it breaks
the "Y is kept as it is" rule). Options for B: `(B, K, 64, 64)` logits with K = 36 / 144 / 256 /
1024. *Filled:* A → `(B, 2, 64, 64)` = (Cb, Cr) in units of ±128 (divide by 128, so the target
is ≈ ±0.13 and the last layer can start near zero); B/B+ → `(B, 256, 64, 64)` logits, a 16×16
grid on (Cb, Cr) — because 36 bins cap the achievable PSNR at 22.31 dB, below the baseline.

**Last layer.** A: options — `Sigmoid` scaled to ±128 (needs the offset: chroma straddles 0),
`Tanh` (bounded at ±128 by construction, harmless here since \|chroma\| ≤ 128 and clipping
happens anyway when the RGB is rebuilt), or **a plain 1×1 conv with no activation**. B: a 1×1
conv to 256 channels, softmax only inside the loss. *Filled:* A → linear 1×1, no activation,
and **Y is written back unchanged** when the RGB is rebuilt (`rgb = Ycc⁻¹(Y, cb, cr)`, then
clamp to [0, 255]); B → linear 1×1 to 256; B+ → the same logits, decoded as
`argmax (log p − γ log p̄)` with **γ = 0.3 chosen on valid** (γ = 0 is B; γ = 1 on this
parameterisation is the 7.9 dB neon disaster of problem 6).

**Training loss.** A: MSE (returns the conditional mean, colourfulness 0.011), **L1**
(0.015, cheaper in gradients and closer to a plausible colour), Huber (no measurable
difference at these scales). B: cross-entropy on the 256 bin labels; B+ is the *same training*
with the prior pull only at inference (which is what the paper does — the important part,
because a γ applied inside the loss trains a different, wrong classifier). *Filled:* A → L1 on
the two chroma planes, plus **one scalar α calibrated on valid** (the α dial of problem 1; the
uncalibrated answer keeps 29% of the colour by construction); B/B+ → cross-entropy over the
256 bins, no class weights in the main run, with the imbalance noted (15.5% of pixels in the
top cell).

**Evaluation metric.** All three, from one function with a written convention, plus the extra
columns the audit says are needed. *Filled:* "**PSNR = mean over photos of 10·log10(1/MSE),
MSE on [0,1] over all three channels, each photo's PSNR clipped at 40 dB; also the median and
the pooled value. chroma L1 = mean over pixels of (|ΔCb| + |ΔCr|)/255. colourfulness = mean
over pixels and channels of |chroma|/255 (= 0.041 on the real photos, 0 for gray). Plus: the
number of photos out of n that beat gray, and the same PSNR split by colour quintile.** For
B/B+ also top-1 bin accuracy — **against its 18.8% ceiling**, never as a headline.

**Baseline.** *Filled:* the gray input (chroma = 0), with the conventions attached: **test
25.36 dB mean (95% CI 24.24-26.56), 23.21 dB median, 24.06 dB clipped at 40, 21.11 dB pooled,
chroma L1 0.0804, colourfulness 0.0000**; valid 26.94 / 23.95 / 24.46 / 21.06. And the two
reference points that make a number mean something: route A's ceiling (conditional mean,
24.36 dB per photo on valid, colourfulness 0.011) and route B's ceiling at K = 256 (the binned
truth, 31.54 dB).

### Things the table cannot hold but the implementation must

1. **The colour transform, written out** (`ycc` / `rgb_of` with the JPEG matrix), plus its
   inverse for the figure grid. `skimage` is not installed in this environment, so
   `rgb2lab`/`rgb2ycbcr` cannot be imported — the two functions are ~6 lines and were
   round-trip-checked to ≤ 0.005 of 255.
2. **Y is copied, never predicted.** If the net's own Y drifts, the PSNR measures the drift
   instead of the colour, and 99% of the signal is Y.
3. **The two calibrated scalars** — α for route A, γ for B+ — chosen on valid, saved, and
   reported with their curves. Without them neither route is colourful, and with them the
   result is partly the calibration's, so say which.
4. **One metric function, one place**, with the 40 dB clip and the definition strings in the
   docstring, used for baseline, all three networks and every figure.
5. **The 8-photo grid, chosen by rule** (colourful / colourless / six across the quintiles),
   as gray | A | B | B+ | truth — and it should include one of the 8 exactly-grey test photos,
   because "leaves black-and-white photos alone" is a behaviour worth seeing.
6. **The two sentences the task asks for**, with the numbers: which network you would pick
   (the one whose photos are colourful at an acceptable cost — read the grid, then the metric
   triple) and what the PSNR would have picked (**the gray baseline: its maximum is at
   α = 0**).
7. **Do not touch the 3325 unused val photos while reporting the 100.** If you use them (which
   problem 7 recommends for the decisions), report both columns: "100 photos, the handout's
   split" and "1000 photos, the unused remainder of `val/`". Never mix them into one number,
   and never let any of them near `test`.
8. **No colour augmentation.** Colour jitter, channel shuffle, autocontrast and solarise all
   edit the thing being predicted; flipping and a small rotation are safe. Worth a comment in
   the code so the next person does not "improve" the pipeline with `v2.ColorJitter`.
9. **A smoke test over 1 batch**: shapes, no NaN, the gray baseline's PSNR reproduced to 0.01
   dB by your own metric function, and a zero-initialised last layer scoring exactly the gray
   baseline (for route A it should — that is the free check that Y is being copied and the
   chroma starts at 0).
10. **The binning, for route B, published as a K × 2 array of cell centres** so the reported
    accuracy and the 31.54 dB ceiling are reproducible from the report alone.
