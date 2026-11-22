# Task 6 — Colourisation from luminance: results, and what was checked

Built to the specification table in cell 77 of `../Copy_of_ex09_student.ipynb`, which is also
`spec.py` and is printed at the top of every log in `results/` before a single number. Two
routes, one shared data definition, one metric module, one look at the test split.

    python run_all.py          89.1 min on 6 CPU threads, seven steps, writes everything below
    python smoke_test.py       16 s, 18 checks, nothing trained
    python spec.py             the specification, plus the settings the run actually used

---

## The bottom line

The 500 test photos, all seven predictors scored by the same `metrics.score` call, the gray
baseline measured here rather than quoted (`results/metrics_test.json`, `results/eval_test_log.txt`):

| predictor | PSNR mean | median | clipped at 40 | pooled | chroma L1 | colourfulness | hue error | beats gray | top-1 bin |
| --- |---|---|---|---|---|---|---|---|---|
| **gray** (the task's baseline) | **25.36** | 23.21 | 24.05 | 21.11 | 0.0813 | 0.0000 | 97.0° | — | — |
| **A** — UNet, L1, α = 1.15 | 24.26 | **24.46** | **24.25** | **21.99** | **0.0744** | 0.0219 | **49.9°** | **61.0%** | — |
| **B** — UNet, 256 bins, γ = 0 | 23.27 | 23.76 | 23.27 | 21.60 | 0.0871 | 0.0339 | 50.1° | 50.2% | **32.69%** |
| **B+** — the same, prior pull γ | 23.27 | 23.76 | 23.27 | 21.60 | 0.0871 | 0.0339 | 50.1° | 50.2% | **32.69%** |
| binned truth (route B's ceiling) | 31.52 | 31.68 | 31.52 | 31.32 | 0.0346 | 0.0505 | 7.0° | 88.0% | — |
| control: chroma shuffled | 22.19 | 22.63 | 22.19 | 20.37 | 0.1014 | 0.0330 | 73.6° | 31.4% | — |
| control: chroma rolled | 22.44 | 22.90 | 22.44 | 20.66 | 0.0973 | 0.0332 | 66.2° | 37.2% | — |

A bold *number* marks the best of the four models in that column (the bold row names are names).
The binned-truth row is
unbolded although it beats all of them: it is not a predictor, it reads the truth's own chroma and
merely quantises it, so its column entries would win every bold in this table while proving
nothing. The controls are unbolded for the same kind of reason — they are not offered as answers.

Five things come out of this table, and they are the whole report:

1. **Gray wins the task's first metric and loses the other three.** On per-photo-mean PSNR gray
   is first (25.36 against A's 24.26). On the median, on PSNR clipped at 40 dB and on the pooled
   PSNR the notebook's own cell 30 computes, A is first - by +1.25, +0.20 and +0.87 dB. Route A
   also halves the hue error (49.9° against gray's 97.0°), has the lowest chroma L1 of any
   predictor that is not the truth, and is the better predictor on 305 of the 500 photos (61.0%).
2. **The mean is a tail, and it is the colourless tail.** Gray's mean sits 2.15 dB *above* its
   own median because 10 of the 500 photos score above 40 dB against it. Those 10 are the
   already-black-and-white ones: 8 of them are exactly grey (`col6_data.exactly_grey`, the audit's
   count of 8 on this split), the other two have a mean chroma radius of 0.97 and 1.08 against the
   split's 15.79, and **7 of the 10 are reproduced bit-exactly, so they score the 120 dB floor**.
   They average 105.08 dB, so between them they put 10 × 105.08 / 500 = **2.10 dB into gray's
   25.36 mean** — 2% of the photos supplying 8% of the number — and without them gray's mean over
   the remaining 490 photos is 23.73 dB. (Those are two different quantities and both are quoted on
   purpose: 2.10 is the ten's share of the average, 25.36 − 23.73 = 1.63 is how far the mean would
   fall if the ten were dropped.) Against A exactly 1 photo clips. Gray's 95% bootstrap interval on
   the mean is [24.24, 26.56], 2.3 dB wide on 500 photos; A's is 0.8 dB wide. The metric the task
   names first is measuring those ten photos as much as it measures the model. (The 105.08 and
   23.73 are arithmetic on the arrays, reproducible by the one-liner at the end of this file; the
   split is checkable against the JSON without running it, since
   (10 × 105.08 + 490 × 23.73)/500 = 25.357, which is the table's `psnr_mean` for gray, and
   `clip40_hits = 10` is stored there too.) The α section measures the same disease on the
   validation split, where it is worse than a weighting complaint: 3 photos supply 3.60 dB of gray's
   26.94 there, and with those 3 removed the metric's own optimum for route A moves from "send no
   colour" to α = 0.55.
3. **B and B+ are the same row because γ was measured at 0.** The published prior correction is
   not free: every γ > 0 spends PSNR to buy colourfulness nobody asked for (`figures/calibration.png`).
4. **Route B beats the ceiling it was quoted against, and that is the expected outcome.** On the
   photos the lookup table is fitted and scored on — the train split, in sample both — route B's
   training loss is 2.1992 nats where the table cannot go below 2.377, and out of sample it is
   32.69% top-1 on test against that table's 30.12%. A table sees one 64-bin luminance number per
   pixel; the network sees the luminance itself. What B *cannot* pass is the other ceiling —
   31.5 dB, the binned truth — because its decoder can only emit bin centres.
5. **Both controls lose to gray**, by 3.17 and 2.92 dB with 95% intervals excluding zero, while
   carrying route B's own chroma moved to the wrong place — not a dull stand-in for colour, since
   their colourfulness is essentially the model's (0.0330 and 0.0332 against B's 0.0339). The
   metric is not paying anyone for merely having colour.

Paired against gray, per photo. The four model rows are `results/metrics_test.json →
paired_vs_gray`; the two control rows are printed by the same step in `eval_test_log.txt` (the
JSON's `paired_vs_gray` holds only the four models, so those two CIs come from the log), and their
win counts are the table's own "beats gray" percentages, 31.4% and 37.2% of 500:

| | Δ dB per photo | 95% CI | sems | wins / 500 |
| --- |---|---|---|---|
| A | −1.10 | −2.24 to −0.08 | 2.0 | **305** |
| B | −2.09 | −3.21 to −1.03 | 3.8 | 251 |
| binned truth | +6.17 | +4.90 to +7.34 | 9.9 | 440 |
| control shuffled | −3.17 | −4.28 to −2.17 | — | 157 |
| control rolled | −2.92 | −4.06 to −1.92 | — | 186 |

A loses the average and wins the majority of photos. That is the same fact as point 2, read off
the other column.

---

## Where each method helps, by how colourful the photo already was

Mean PSNR change against gray, per quintile of the photo's own colourfulness
(`figures/quintiles_test.png`, `metrics_test.json → quintiles`):

| test quintile | mean chroma | A | B / B+ | binned truth |
| --- |---|---|---|---|
| Q1, dullest | 4.4 | −9.99 | −12.57 | −8.81 |
| Q2 | 9.5 | −0.24 | −1.21 | +5.29 |
| Q3 | 13.8 | +1.09 | +0.65 | +8.49 |
| Q4 | 19.0 | +1.56 | +1.14 | +10.98 |
| Q5, most colourful | 32.3 | +2.09 | +1.56 | +14.88 |

Every coloured predictor loses on the dullest quintile - and even the binned truth, which is
given the answer and allowed only to round it, loses 8.8 dB there. Above Q2 every predictor
gains. This is the audit's claim, measured on the test split: on a photo with no colour to find,
an average colour is a damage, and the damage is larger than the gain everywhere else. It is why
the task's mean picks gray and nothing else does.

The quintile rows are an equal-size partition — five groups of n = 100, printed in the JSON — of
the same 500 photos, so their deltas must average back to the paired difference of the table above.
They do: (−9.99 − 0.24 + 1.09 + 1.56 + 2.09)/5 = −1.098 dB, which is A's paired mean to every
printed digit, and the same arithmetic on the other two columns gives −2.086 (B, exact) and +6.166
against the paired table's +6.168 — a 0.002 dB gap, which is the rounding of five numbers printed
to 2 dp and nothing else. `metrics.quintiles` and `metrics.paired` are two functions fed the same
per-photo arrays, so this is one dataset agreeing with itself through two routes, and a reader can
do it with a calculator.

`figures/photos_test.png` shows the same thing as pictures. Its seven rows are the ones rule ET5
names in `results/eval_test_log.txt`, in that order — 164 (most colourful: a golf ball on grass
under blue sky), 210 (among the dullest: a stone church), 20 (already black and white: a shed),
then the seeded draws 135 (a running spaniel), 256 (a man holding a ball), 423 (a garbage truck)
and 318 (a mouse with a French horn). Three things are visible without the table: on the ball, the
dog and the truck A puts green where green belongs; on the shed, which has no colour to find, A
and B both lay colour onto a photo that does not have it; and on the sky in row 1, B and B+ show
pale horizontal bands, which is the 16×16 bin grid becoming visible — a bin-centre decoder cannot
render a gradient finer than its own cells, and that is the same 31.5 dB wall as a texture.

---

## Every choice, measured on the 100 validation photos

Nothing in the models or the metrics was set by hand. The sweep is 29 arms and 49.0 min of step
wall clock, inside which `choices.json → totals` accounts 2935.0 s of arm time — 29 arms,
`quick: false`, 6 threads, the two agreeing to within seconds. The arms are in
`results/sweep_log.txt`, under rules R1–R5 written in `validate_choices.py`'s docstring before the
first arm ran:

| | route A | route B |
| --- | --- | --- |
| widths | `[16, 32, 64]` (117,410 parameters) | `[16, 32, 64]` (121,728 parameters) |
| learning rate | 3e-3 | 3e-3 |
| batch size | 64 | 64 |
| horizontal flip | on (arm measured, not assumed) | **off** (arm measured: on is worse by 0.034 nats) |
| pixels per photo in the loss | all 4096 | 1024 of 4096 |
| epochs trained / epoch kept | 16 / **14** (valid L1 0.07129) | 16 / **16** (valid CE 2.15919) |
| wall clock | 974.5 s | 1191.0 s |

Rule R1 is the one that matters most and it is a negative rule: **arms are ranked by validation
loss, never by PSNR**, because per-photo PSNR on this data is maximised by predicting no colour
and a PSNR-ranked sweep would rank models by how dull they are.

**What the seed can move.** Stage 3 re-ran the chosen configuration at three seeds. Best
validation loss sd: route A 0.00088 (1.15% of the loss), route B 0.01350 (0.57%). Valid
colourfulness moves 16.6% of its own value on route A from the seed alone. Rule R4 says a
difference smaller than 2 of those is reported as a tie, and `python validate_choices.py
--report` (no training, reads `results/choices.json`) prints the verdict on every arm:

* route A stage 2 - **all three** other batch/flip arms are inside 2 sd of the winner (the
  farthest is 0.00131 away and R4's floor is 0.00176). The flip is worth 0.00131 L1 at batch 64
  and 0.00058 at batch 32; the seed moves 0.00176. Kept because it is better and free, not
  because it was shown to matter.
* route B stage 2 - flip-on is **outside** 2 sd (+0.0343 nats), so route B's flip really does
  hurt; the batch size does not (b32 is +0.0196, inside the band).
* the pixel count - 512, 1024 and all 4096 pixels are all inside 2 sd of each other (+0.0070 and
  +0.0009 nats), so the choice is decided on cost: 19 s/epoch instead of 33. `smoke_test.py`
  proves the sampled loss is the same objective as the full one (max difference 1.49e-07).

Because `pick` asks an arm to be inside **both** R4's 2-sd floor and R3's 0.2%-of-loss
indifference band, the tighter of the two governs which arm survives, and on both routes that is
R3's. R3's band is 0.2% of the *pool's own* best loss, so it is a different number in each pool:
0.00015 in both of route A's (bests 0.07613 and 0.07501) and 0.00458 to 0.00464 across route B's
three (bests 2.28967, 2.29639, 2.31804). R4's floors are 0.00176 and 0.02700 — five to twelve
times wider — so R4 decides the *wording* of the report and R3 decides the *arm*. `--report` prints
one band per route (0.00015 and 0.00458), computed off stage 2's best arm rather than each pool's
own, which is a slightly different quantity and the reason it is not the number in the closest call
below. In the four main pools R3's band admitted the winner alone and the size/cost tie-break never
fired (closest call: `1/B w32-64-128 lr0.003` at +0.00477, which misses that pool's own band of
0.00464 by 0.00013). It fired exactly once, in the pixel pool, where `pixels1024` and `pixelsall`
both fell inside both bands and the tie went to 19 s/epoch over 33 s/epoch. Worth saying plainly: a
sweep whose differences are mostly inside the seed's reach is a sweep that mostly did not matter,
and the report prints which ones did.

---

## α: one scalar, and six defensible answers to "which one"

Route A's output is scaled by a single number, calibrated on valid by rule CA: maximise
per-photo mean PSNR subject to colourfulness ≥ half the photos' own. That gives **α = 1.15**,
24.56 dB at colourfulness 0.0209 against the photos' 0.0402 (`results/calibration.json`).

The same table answers the same question six ways, and the answers are not close:

| rule for choosing α | α | valid PSNR | colourfulness |
| --- |---|---|---|
| **CA (used): PSNR subject to the colourfulness floor** | **1.15** | 24.56 | 0.0209 |
| unconstrained per-photo PSNR | **0.0** | 26.94 | 0.0000 |
| pooled PSNR | 1.00 | 24.79 | 0.0182 |
| PSNR clipped at 40 dB | 0.50 | 25.20 | 0.0091 |
| minimum chroma L1 | 0.80 | 25.01 | 0.0145 |
| best hue accuracy | 1.70 | 23.50 | 0.0306 |

(The PSNR column is the per-photo mean at that α, whatever rule did the choosing; `psnr_pooled`
at the pooled row's α = 1.00 is 22.03 dB, and the whole 41-point curve is in the JSON.)

Six conventions give six α spanning the whole range from no colour to over-saturated; the honest
claim is only that α was chosen by a rule stated before it was measured.

α = 0 is "send no colour", and it is the argmax of the metric named first in the task text. Two
things produce that, and they are worth separating: the data, which the audit measured (these photos
carry so little recoverable colour that a wrong guess costs more than a good one earns), and the
metric's own floor, which the two paragraphs below attribute to individual photos. Both are real,
and the second one needs no model to see: `python metrics.py` prints that a perfect prediction
scores `10·log10(1/EPS)` = 120.0 dB, "that is the epsilon, not the model". A rule with a
colourfulness floor is what the first one needs.

The α curve shows the same fragility from the other side, and here it can be attributed to named
photos rather than asserted as a mechanism. PSNR is 26.94 dB at α = 0, **25.32 dB at α = 0.05** — a
1.62 dB cliff in a single step of the grid — and then holds between 25.22 and 25.32 all the way to
α = 0.4. Three photos are the cliff. The validation split contains 3 photos that are exactly black
and white (`col6_data.exactly_grey`: R == G == B in every pixel, chroma radius 0.00), the gray
baseline reproduces them bit-exactly, and at α = 0 they therefore sit on the 120 dB floor that
`metrics.EPS` defines. Between α = 0 and α = 0.05 their mean falls to 61.55 dB — **−1.75 dB of the
split mean on its own — while the other 97 photos gain +0.13 dB.** At α = 0 those three supply 3.60
dB of gray's 26.94: 3% of the photos, 13.4% of the number. (The test split's version of this is
finding 2: ten photos, 2.10 dB of 25.36.)

Then run the same argmax question with the three floor-bound photos removed:

| mean PSNR taken over | argmax α | at that α | at α = 0 |
| --- |---|---|---|
| all 100 validation photos | **0.00** | 26.94 | — |
| the 97 that are not on the floor | **0.55** | 24.70 | 24.06 |
| the 3 exactly-black-and-white ones | 0.00 | 120.00 | — |

So "PSNR says send no colour" is not, on this data, a statement about colourisation. It is a
statement about three photos whose PSNR is a property of a float epsilon: with them removed the
metric's own optimum moves to α = 0.55 and rises monotonically to it. Two consequences, both
reported rather than smoothed over. The plateau is not photos trading colour for PSNR at par — the
97 gain +0.59 dB between α = 0 and α = 0.4 while the all-100 mean *falls* 1.72 dB, and the flatness
is that gain being cancelled by the tail's decay. And the colourfulness floor is still doing real
work after the artifact is removed: α = 0.55 reaches only 0.0100 against the CA floor of 0.0201, so
even the de-artifacted metric prefers a duller picture than rule CA allows. This is arithmetic on
the arrays and on `results/model_A.pt`, not a shipped JSON, reproducible by the second one-liner at
the end of this file.

Two more things from the same curve. **α is a magnitude knob only**: the hue error is 45.4° at
α = 0.05 and 45.2° at α = 2.0, so the whole usable range moves hue by 0.2°,
and two of the six rules above ("best hue accuracy" at 1.70, "minimum chroma L1" at 0.80) are
argmaxes taken over a line that is flat to within noise. And the curve's α = 1.00 row — 24.79 dB,
chroma L1 0.0715, colourfulness 0.0182 — is *identical* to `train_log.json`'s own
`valid_uncalibrated` block: step 4 scored these same weights at α = 1 and wrote that number into
`train_log.json` before step 5 loaded the checkpoint at all. That the two agree says the checkpoint
round-tripped through the disk and the calibration sweep started from the weights it claimed to,
which is the cheapest consistency check in this folder.

## γ: measured, and it is zero

Route B's inference-time prior pull, `argmax(log p − γ log p̄)`, was calibrated by the same rule.
**The measured answer is γ = 0.** The full curve is in `results/calibration.json`: γ = 0.1 costs
0.028 dB for 0.0008 of colourfulness, γ = 0.5 costs 0.707 dB, and γ = 0.85 (the best hue accuracy)
costs 8.79 dB and reaches colourfulness 0.1137 — 2.8× the photos' own — while its bin accuracy
*falls* from 31.66% to 16.92%. So **B+ is B**, in the table, in the figures and in the numbers, and
`figures/chroma_support_test.png` draws them as the same blob. That is what γ = 0 means, and it is
a measured outcome, not a skipped arm.

The audit's γ = 0.3 is not contradicted: it was chosen against a different objective (it needed
a colourfulness floor to survive at all, which is what rule CA supplies here too).

---

## Ceilings: which of these numbers is a wall

Three ceilings were measured before any model existed and are re-measured by `smoke_test.py` on
every run, so a claim about one can never drift from the grid it was computed on:

* **31.54 dB — the binned truth at 256 bins**, i.e. the truth quantised onto the same 16×16 grid
  route B decodes to. This is a **wall**: no decoder that can only emit bin centres can pass it.
  Route B lands 8.25 dB under it on the test split. It measures 31.54 dB on valid, 31.52 dB on
  test and 31.55 dB on the 3325 unused val photos - the same number on three splits, which is
  what a property of the grid rather than of the sample should do.
* **30.12% top-1 and 2.377 nats — what a colour lookup table achieves** at the 16×16 grid
  (`metrics.ceilings`: chroma read off one 64-bin luminance number per pixel). Both are measured
  **on the 9469 training photos the table is fitted on — in-sample**, which flatters the table.
  It is a **fence around one method**, not around the task: the better-known **18.77% and
  3.343 nats** are the same table at the audit's 32×32 grid, and quoting either of them as a bound
  on a network would be wrong, because a network sees full-resolution luminance and a
  neighbourhood around each pixel and is entitled to do better. It did, and the matched comparison
  is the one on identical photos: route B's training loss at its last epoch is **2.1992 nats on the
  same train split where the table scores 2.377**. Out of sample the same ordering holds —
  **31.66% top-1 on valid and 32.69% on test**, against a table that reached 30.12% while being
  scored on its own training data. The bottom table prints the ceiling beside the accuracy, and
  both metrics JSONs store it as `lookup_accuracy_ceiling_pct`, so the pairing is never left to
  the prose.
* **gray's 25.36 dB** — not a ceiling at all, but the number every row is compared against, and
  it is re-measured per split by the code being judged (valid 26.94, test 25.36, valrest 25.58;
  the audit's test number was 25.36, reproduced to 0.01 dB).

---

## The epoch budget, and what selecting on 100 photos actually buys

16 epochs was fixed in the specification, not swept (a sweep at a quarter of the data would pick
a different epoch count from the full run). The epoch kept is the best validation loss, per route.

Both train curves fall monotonically (A 0.0809 → 0.0740, B 2.4198 → 2.1992, every epoch lower than
the last). Both *validation* curves are noisy enough that the kept epoch is a soft statement:

* **Route A** flattens by epoch 4–6 and then oscillates: valid L1 0.0746, 0.0757, 0.0738, 0.0739,
  0.0760, 0.0764 over epochs 4–9. The scatter of its last eleven validation points is sd 0.00175 -
  **twice the 0.00088 seed sd** - around a flat level, and epoch 14's 0.07129 is the bottom of that
  scatter. Its valid PSNR at epoch 14 is 24.79 dB against 24.58 dB averaged over epochs 6–16.
* **Route B**'s minimum is its last epoch (2.15919), and its train loss is still falling, so the
  budget did bind - but its validation points scatter with sd 0.0624, **4.6× the 0.01350 seed
  sd**, including a 2.3853 at epoch 12 that sits 0.18 nats above both of its neighbours. Epoch 16's
  2.15919 is 0.33 dB of valid PSNR above the mean of the last eleven checkpoints (23.424 against
  23.098 - three decimals, because the two printed to two of them subtract to 0.32).

The consequence is a bias, and it goes one way: taking the argmin of 16 checkpoints scored on 100
photos reports a *selected* validation loss, not an unbiased one - here by roughly 0.2 dB (A) and
0.3 dB (B) of valid PSNR. The test numbers are not affected, because the test split played no part
in which epoch was kept; the validation numbers quoted in this file are the ones to read with that
0.2–0.3 dB in mind, and α = 1.15 was calibrated on the epoch-14 weights so it inherits the pick.
What is *not* noise is the direction: every checkpoint from epoch 6 to 16 puts route A's valid PSNR
inside a 0.51 dB band (24.28–24.79) and route B's inside 0.95 dB (22.48–23.42), and route B's train
loss at the last epoch (2.1992, still below every epoch before it) says it would have used a longer
run.

---

## The split discipline

* **train, 9469** — everything is fitted here (the sweep uses a fixed 2367-photo subset of it,
  the same photos in every arm, rule R5).
* **valid, 100** — decides widths, learning rate, batch size, flip, pixel count, α, γ, and the
  epoch kept. Every choice in this folder was made here.
* **valrest, 3325** — the validation photos the handout leaves unused. Nothing was chosen on it.
  It is scored once, as a confirmation column: gray 25.58, A 24.39 (beating gray on 59.8% of
  3325, against 61.0% on the 100), B 23.36, binned truth 31.55. Agreement on 33× the sample is
  what makes the 100 defensible rather than lucky.
* **test, 500** — read once, by step 7 of `run_all.py`, by `evaluate_task6.py`, and by nothing
  else. `results/eval_test_log.txt` is that look. `run_all.py --until 6` runs everything before
  it, and no number in this file feeds back into a model.

The one exception to "decides nothing" is procedural and is stated in `evaluate_task6.py`'s ET1:
nothing after the test scoring reads its output. The results JSON is written *before* the figures
are drawn, so a plotting failure costs a figure and not the results file.

---

## What convinces us there are no mistakes

This is the part the report exists for. Six independent layers, in the order they ran.

**1 · The 18 checks, before anything was trained** (`python smoke_test.py`, 16 s,
`results/smoke_log.txt`). Each is a claim the specification makes, turned into an assert with a
printed number beside it. The ones that would catch a silent, plausible, wrong pipeline:

* `metric == the audit's metric` — gray on valid: 26.94 mean, 24.45 clipped at 40, 21.06 pooled,
  23.95 median, chroma L1 0.0804, each within 0.02 of a number measured in
  `../task6_data_audit.md` **before this code existed**. A chroma channel swapped, a scale
  applied twice, or a different PSNR convention cannot produce these.
* `zero-init route A == the baseline` — with a zero-initialised head the model scores the gray
  baseline to **0.00e+00 dB** on every photo, which is a free proof that Y is written back
  unchanged and that the ±128 scale is applied exactly once.
* `Y is copied, not predicted` — with a deliberately non-zero head pushing |chroma| to 11.6 of 128
  (colourfulness 0.023, i.e. a real colour output), the decoded luminance still equals the input Y
  to 3.05e-05 of 255 on the 95.5% of pixels that stay in gamut, and the other 4.5% move only where
  the clip legitimately bites. Non-vacuous in both directions.
* `route-B init == the prior` — the 256-bin head initialised with bias = log(train marginal) has
  cross-entropy 2.5501 nats = H(C) exactly (not log 256 = 5.545, which a zero-bias init would
  give), constant across pixels and photos to 0.0e+00.
* `sampled head == full head` — the 1024-of-4096 pixel loss route B trains on is the *same*
  function as the full one (1.49e-07 over 2341 pixels of a random model), so the pixel-count arm
  measures cost, not a different objective.
* `the bin grid` / `ceilings` — the 16×16 grid tiles (Cb, Cr) ∈ [−128, 128)² into 16-unit cells and
  every cell centre is within 8.0 of its pixels in each coordinate - half a cell, by construction
  (`step / 2`, asserted not observed); the binned truth scores the audit's 31.54 dB; the
  lookup-table accuracy reproduces 18.77% at 32×32 and gives 30.12% at the 16×16 grid this folder
  actually trains (193 of its 256 cells populated). Every ceiling quoted above is recomputed by
  that script, on this run.
* `control: shuffled` / `control: rolled` — on valid, against gray's 26.94, the truth's chroma
  relocated at random scores 20.41 dB (a paired deficit of 6.53 dB, 95% CI 3.51 to 10.30, losing
  83 of 100 photos) and the same chroma rolled by half the image 25.74 dB (−1.21 dB, CI 0.70 to
  1.67, losing 74). Both keep the truth's chroma — a spatial permutation moves the Hasler–Süsstrunk
  statistic a little (0.0374 and 0.0379 against the truth's 0.0402) and nothing else. A metric that
  rewarded colour as such would have failed here. `evaluate_task6.py` runs the same two controls on
  the *model's* chroma instead of the truth's, on both splits — a milder case, since the model's
  chroma is already duller than the truth's, and they still lose (top of this file: −3.17 and
  −2.92 dB on test, −4.94 and −4.56 on valid).
* `gamma is inference-only` — on a constructed `log p` the pull demonstrably changes decisions
  (towards *rare* bins, which is the direction the published correction actually goes) while
  leaving all 76 parameter tensors bit-identical.
* `every call binds` — 240 call sites across the folder's 13 pipeline scripts are checked against
  the real signature of the function each names, and every `name.attribute` against the imports the
  file actually has. (Thirteen means *steps in the run*: the check globs `*.py` and skips the
  underscore-prefixed tools beside them, so that the number a log printed cannot drift just because
  an after-the-fact tool was added to the folder.) Neither check is theoretical: `figs.legend_on`
  was declared with a leading argument no call site passed, and `train_task6.py` only said so two
  minutes in, after it had trained route A and written half its output.
* `the figures draw` — the three plotting functions of `evaluate_task6.py` are executed here on 8
  synthetic photos into a temporary directory, because in the real pipeline they are the last
  thing to run after the one look at the test split, and a `NameError` there would have cost the
  run its results file. (`figs.FIGURES` was exactly such a `NameError`: found by this check on
  its own first execution, after everything else had already passed.)

**2 · Rules written before the arms, and constants pinned to the prose.** R1–R5 in
`validate_choices.py`, CA–CE in `calibrate.py`, ET1–ET5 in `evaluate_task6.py`, all in docstrings,
all printed with the results. Where a rule decided something a reader might dispute, the alternative
is reported too: all six α, all six γ (their own `alternatives` block, straight out of
`results/calibration.json`), and the R3-vs-R4 verdict on every one of the 29 arms. Beside them,
`spec.py` **asserts** that four of the numbers its own table states still equal the code's constants
— grid 16, 256 bins, ±128 chroma units, the 40 dB clip — and prints seven of them next to the file
and symbol each came from (`metrics.EPS 1e-12`, `metrics.DRAWS/SEED (2000, 0)`, …). Every script in
the folder calls it before it prints a single number, so prose drifting away from code stops the run
instead of sitting in a comment.

**3 · Two PSNR conventions printed side by side everywhere.** On this data they rank the
predictors in opposite directions. One number alone would have produced a confident, wrong
headline in either direction.

**4 · Controls, on both splits, and a third time with no data at all.** Same code path, same
baseline, wrong colour: they lose on valid, they lose on test, and they lose on the four synthetic
photos of `python metrics.py`'s self-test (gray 18.68 dB against shuffled 15.94 and rolled 15.90),
which is the version of the check that cannot be accused of a split-specific accident.

**5 · The confirmation column.** 3325 photos that decided nothing reproduce the 100-photo
ordering and the 61% beat rate to within a point.

**6 · Plumbing proved before it was expensive.** Four commands were run end to end on 2-epoch
weights, ~10 min, before the 90-minute run spent its time on 16:

```bash
python train_task6.py --epochs 2            # -> results/dryrun_train_log.txt        (6.2 min)
python calibrate.py                         # -> results/dryrun_calibrate_log.txt    (0.8 min)
python evaluate_task6.py --valid            # -> results/dryrun_evaluate_log.txt
python evaluate_task6.py --valid --valrest  # -> results/dryrun_eval_valid_valrest_log.txt
```

The last one exists because the 3325-photo confirmation is the only block a `--valid` run skips and
the test run executes, so it would otherwise have been executed for the first time on the one look
that cannot be repeated. What these four logs prove is that the whole path survives real weights —
decode, calibrate, score, plot — for ten minutes instead of ninety. What they are not: they are not
part of the shipped run's verified chain. They come from the rehearsal's throwaway weights, so
every number in them is a rehearsal number — its α is 1.4 where the report's is 1.15, and its
route B scores 22.89 dB on the validation photos where the shipped 16-epoch model scores 23.42 on
those same 100 — they predate the last edits to `smoke_test.py` and
`run_all.py`, and the byte-for-byte assembly assertion below covers the seven step logs, not these.
`results/calibration.json` and `results/metrics_valid_dryrun.json` beside them are the real ones:
the rehearsal wrote its own copies first and steps 5 and 6 overwrote them, which is why the
rehearsal has to happen before the run and not after it. That second filename is a wart worth naming
rather than explaining away: `evaluate_task6.py` sends every `--valid` run to
`metrics_valid_dryrun.json`, the rehearsal's and step 6's alike, so the shipped file's contents are
step 6's (`"alpha": 1.15`, `"seconds": 7.7` — the real calibration and a 7-second scoring, not the
rehearsal's 1.4 and 3.8 min) while the word "dryrun" in its name is the flag's wording, not a claim
about what is inside. `run_all_log.txt`'s step-6 header reads `evaluate on valid (dry run)` — the same
wording, over a step that really does score the 100 validation photos and really is not allowed to
decide anything. In this folder "dry" describes what a step may change, never what it reads.

**The two bugs the process caught, written down because they are the evidence the process
works.** `figs.legend_on`'s phantom argument (above) reached a trained model before it was seen;
after it was fixed, the binding check was widened to cover bare module aliases, and the alias
check immediately failed on `figs.FIGURES` in the smoke test itself. Separately, rule R4's
override path was found to print "the R4 pick is used" while writing the R3 pick's knobs, and
`pick`'s cost tie-break was counting an arm as cheaper for having trained fewer epochs. Both are
fixed; neither changed this run, and that is checkable rather than asserted - `agree: true` for
both routes in `results/choices.json`, and replaying the corrected `pick` over the arms recorded
there returns `2/A b64 flip1` and `2/B b64 flip0`, the two configurations the trained models were
built with.

---

## How to check this yourself

```bash
python -m venv .venv && . .venv\Scripts\activate && pip install -r requirements.txt

# Step 1 finds the photos itself: it looks in data/imagenette2-160, then in the repo's
# ../_imagenette_dl/imagenette2-160, and otherwise downloads the 94 MB archive - but it
# accepts a copy only if it holds exactly 13394 JPEGs (col6_data.locate). To place one:
curl -LO https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
tar -xzf imagenette2-160.tgz && mv imagenette2-160 data/imagenette2-160

python smoke_test.py            # 16 s: the 18 checks, nothing trained
python run_all.py --until 6     # ~87 min, everything except the one look at the test photos
python run_all.py               # +2.5 min more: that look, and the figures drawn from it
```

What this run was made with: Python 3.13.2, `numpy==2.5.5`, `torch==2.15.0` (CPU),
`matplotlib==3.11.2`, `Pillow==13.0.0`, Windows, 6 torch threads. `requirements.txt` pins the four
libraries, and the thread count is an artifact rather than a sentence — `fitting.threads(6)` is the
one `torch.set_num_threads` call in the folder and its argument is recorded in `choices.json`,
`train_log.json`, `calibration.json` and both metrics JSONs. Two of the five versions are recorded
too — `train_log.json` carries `torch: 2.15.0+cpu` and `numpy: 2.5.5` from the modules it imported,
and `choices.json → totals` the numpy one — but **`matplotlib`, `Pillow` and the Python version are
in no artifact**, so those are this paragraph's claim and not a log's. The command below prints
whatever the machine you are reading this on actually has, which is the comparison worth making. A
CPU-vs-CUDA change is not cosmetic — the seconds here are 6-thread CPU seconds, and
`requirements.txt` says so next to the torch line.

```bash
python -c "import sys,numpy,torch,matplotlib,PIL; print(sys.version.split()[0], numpy.__version__, \
torch.__version__, torch.get_num_threads(), matplotlib.__version__, PIL.__version__)"
```

Then, without retraining anything:

```bash
python spec.py                                # the specification, plus the settings as built
python validate_choices.py --report           # rule R4 on the arms already in choices.json
python metrics.py                             # the metric's own self-tests and the ceilings
python _compare_runs.py                       # this run's results/ against the previous run's
```

`_compare_runs.py` is the second end-to-end run made audible: it walks every field of every
`results/*.json` against a second directory and prints either "IDENTICAL" or the field that moved,
listing the fields that *are* wall clock separately instead of quietly skipping them — `seconds`,
`started`, `secs`, `elapsed` and anything ending `_seconds`, because `epoch_seconds` is sixteen wall
clocks and counting those as numeric would make the tool report disagreement about a machine that
happened to be busy. Logs are compared line by line the same way, so a line differing only in a
"57.3 s" is the same line and a line differing only in a temporary directory's random name is
labelled as one rather than silently forgiven. It retrains nothing and contributes no number of its
own. The default second directory is `_prev_run/results/`, the earlier run on this machine, which is
not part of what you clone — to make the same claim yourself, run `run_all.py` twice with the first
`results/` moved aside and pass the two directories as arguments. (The leading underscore is what
keeps this file out of the binding check's script count: that count means *pipeline* steps.)

**These artifacts are the second run of two, and the two agree.** Run #2 exited 0 in 89.1 min —
sweep 49.0, training 36.2, calibration 0.8, the two evaluations 0.2 and 2.5, data and smoke 0.4 —
and on the console, after closing the combined log, `run_all.py` printed the one line that cannot
live inside the file it counts:

    log assembly verified: the 7 per-step logs appear in run_all_log.txt in order, byte for
    byte - 55517 of its 58274 characters, the rest being this script's own headers and summary

Against run #1, `_compare_runs.py` prints `VERDICT (the numbers): every field of every JSON is
identical, except the ones that are durations or timestamps`. That is 29 arms' validation losses,
sixteen epochs of train and valid loss on both routes, the seed spread, every quintile, every
bootstrap interval and every ceiling — `choices.json` reports 0 numeric and 168 wall-clock
differences, `train_log.json` 0 and 33, `metrics_test.json` 0 and 0. `data_log.txt`,
`eval_valid_log.txt` and `eval_test_log.txt` came out byte-identical; `smoke_log.txt` differs only
in the `mkdtemp` directory name its figure check prints. Three logs differ by more than wall clock,
and their difference is the one genuinely instructive thing about the pair of runs: the `AS BUILT`
header, which `spec.text()` fills from whatever artifacts are already on disk. Run #1 started with a
rehearsal's `choices.json`, `train_log.json` and `calibration.json` still in `results/`, so its step
3 printed `alpha (route A) 1.4` and `A: 2 kept of 2 trained` *before it had swept anything* — a
throwaway rehearsal's answers, presented as the run's own. The shipped run started from an empty
`results/` (run #1 moved to `_prev_run/`), so its step-3 header carries no `AS BUILT` block at all —
88 lines against run #1's 103 — and its step-4 header says `budget 16, not trained yet` and `not
calibrated yet`. No number moved, because that header is printed and never read, and no step reads
an artifact older than the step that wrote it; the agreement above is the proof. The habit is in the
report anyway: empty `results/` before a run, since a header that quotes the disk is only honest
when the disk is empty.

Two sets of numbers in this file are not in a `.json`: the black-and-white decomposition in finding
2 (105.08 dB for the ten clipped test photos, 23.73 dB for the other 490), and the α attribution in
the α section (the −1.62 dB cliff split into −1.75 and +0.13, the 120 → 61.55 dB fall, the α = 0.55
argmax over the 97). Both are arithmetic on the arrays — the second one also reads the trained
weights — and both are reproducible:

```bash
python -c "import numpy as np, col6_data as C, colorspace as CS, metrics as M; \
d=C.build(verbose=False); rgb=C.arrays(d,'test'); p=M.psnr_photos(M.baseline_photos(rgb),rgb); \
o=p>40; print(o.sum(), (C.exactly_grey(rgb)&o).sum(), p[o].mean(), p[~o].mean())"
# 10 8 105.078636 23.729654      <- what that prints here, 21 Nov 2026
```

```bash
python - <<'PY'          # the alpha cliff, attributed; needs results/model_A.pt
import numpy as np, col6_data as C, colorspace as CS, fitting, metrics as M
d = C.build(verbose=False); v = C.arrays(d, "valid"); g = C.exactly_grey(v); n = len(v)
y = CS.ycc(v)[0]; one = fitting.decode(fitting.load_model("results/model_A.pt")[0], v, alpha=1.0)
per = lambda a: M.psnr_photos(CS.rgb_of(y, one["cb"] * a, one["cr"] * a), v)
dd = per(0.05) - per(0.0)
print(f"{int(g.sum())} exactly-grey photos; cliff {dd.mean():+.3f} dB = grey {dd[g].sum()/n:+.3f}"
      f" + other {dd[~g].sum()/n:+.3f}")
print(f"the grey three: {per(0.0)[g].mean():.2f} dB at alpha 0, {per(0.05)[g].mean():.2f} dB at 0.05")
a = [round(float(x), 2) for x in np.arange(0, 2.001, 0.05)]
m = np.array([per(x)[~g].mean() for x in a])
print(f"argmax over the {int((~g).sum())} non-grey: alpha {a[int(m.argmax())]} at {m.max():.2f} dB,"
      f" vs {m[0]:.2f} dB at alpha 0")
PY
# 3 exactly-grey photos; cliff -1.622 dB = grey -1.754 + other +0.132
# the grey three: 120.00 dB at alpha 0, 61.55 dB at 0.05
# argmax over the 97 non-grey: alpha 0.55 at 24.70 dB, vs 24.06 dB at alpha 0
```

Everything else is in a `.json` under `results/`. So is the provenance of every log there: the 13
`.txt` files are 7 step logs written by `run_all.py`, the `run_all_log.txt` they were assembled
from, 4 `dryrun_*` logs from the 2-epoch rehearsal described above, and `choices_report_log.txt`
from `python validate_choices.py --report`. Each one is the console output of exactly one command,
and the seven and the combined file come from one stream — `run_all.py` writes both, and its last
act is to assert that the seven appear inside the combined log, in order, byte for byte, printing
how many of its characters they account for. That assertion is why no step log can be a leftover
from a different run, which is the mistake an earlier draft of this sentence papered over: the
shipped `smoke_log.txt` had once come from a later execution than the combined log it sat inside.

---

## Honest limits

* **Neither route beats gray on the metric the task names first.** A loses it by 1.10 dB
  (2.0 sems) while winning 305 of 500 photos, the median, the clipped mean and the pooled mean.
  If the graders' headline is the per-photo mean, gray is the winner and this report says so in
  its first table.
* **The task's headline metric is the wrong summary for this data**, and the cause is half the data
  and half one constant in this folder's own code. 8 of the 500 test photos are *exactly* black and
  white and two more have a mean chroma radius under 1.1 where the split's is 15.8, so gray is
  near-perfect on ten of them — seven exactly, at the 120 dB floor — and a mean over photos lets
  those ten carry 2.10 dB of its 25.36. The floor is `metrics.EPS = 1e-12`, so the perfect ones are
  a decision this folder made, not a fact about the photographs — `spec.py` prints it beside the
  symbol it came from, and `python metrics.py` prints 120.0 dB out of a synthetic perfect
  prediction — and the clipped mean exists because of it. Median, clipped mean and pooled mean are
  all defensible and all favour colour; the choice among them is a convention, stated rather than
  defended. The strongest form of the objection is measured in the α
  section rather than asserted here: remove the 3 floor-bound validation photos and route A's PSNR
  argmax moves from α = 0 to α = 0.55.
* **Route B's budget was binding.** Its best epoch is its last epoch. A longer run would improve
  it and the comparison with A would shift; the numbers here are for 16 epochs, stated in
  `spec.py`'s "as built" block.
* **α and γ are calibrated on 100 photos**, and colourfulness itself moves 16.6% of its value
  from the seed on route A. α = 1.15 is a rule's output on a small sample, not a physical
  constant; the α curve being flat from 0.05 to 0.4 softens this only for the dull end.
* **One seed for the final models.** The sweep measured the seed spread instead of paying for
  three final runs, and quotes it beside every choice.
* **64×64 centre crops of imagenette**, the notebook's own data definition: a median 75% of the
  resized photo, and the un-prefiltered BICUBIC 160→64 shrink puts a 42.8 dB floor under the fine
  colour detail any model here could ever show (`../task6_data_audit.md`, section 8).
* **Top-1 bin accuracy is a weak metric and is never used to claim anything** - it is printed
  beside its lookup-table ceiling and its per-photo median (32.69% pooled, 31.01% per photo) and
  never as a headline.
