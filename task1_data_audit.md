# Task 1 (LUDB ECG delineation) — audit of the data cell

What the data actually is, measured on all 200 records (PhysioNet LUDB 1.0.1,
lead II, exactly what the notebook's data cell loads), and what to do about it.

Figures: `task1_audit_figs/fig1_unlabelled_edges.png`, `fig2_record104.png`,
`fig3_class_balance.png`. Repaired data cell: `task1_data_cell_fixed.py`.

---

## 1. The labels stop before the signal does — 27% of the samples are not labels

LUDB annotates only the middle of each 10 s record. Measured over the 200 records:

| | |
|---|---|
| median annotated span | 3640 samples = **7.3 s** of the 10 s (shortest 4.6 s) |
| first annotation | median sample 659 (1.3 s in) |
| last annotation | median sample 4321 (8.6 s) |
| samples outside the span, labelled `background` by the cell | **26.9%** |
| records with at least one beat outside the span | **195 / 200** (median 2, up to 6) |
| unlabelled beats total | **413**, against 1817 labelled QRS — about 1 beat in 5 |

This is a property of the released database, not of the notebook's reading of it:
the raw annotation files themselves contain no symbol past sample 4814 in any
record, and their wave count (4874 peaks per lead) matches the README's
58429/12 ≈ 4869. It is also exactly what the task text hints at — *"Look at the
plot closely, also at the start and the end of the record."*

**Those edge regions are not empty.** A plain detector (squared derivative,
smoothed over 120 ms, `find_peaks`, `distance=175`, `prominence=0.08`) was
calibrated on the labelled span: it recovers 1782/1817 = 98.1% of annotated QRS,
and produces only 61 extra peaks there, with median amplitude **1.2 σ**. In the
unlabelled edges it finds 413 peaks with median amplitude **5.8 σ** — the same as
a real labelled beat (5.9 σ) — and 87% of them are at least half as tall as a
typical beat. They are heartbeats, not noise.

### Why this costs you the task

`y = np.zeros(...)` then `y[onset:offset+1] = c` means *unannotated = background*,
so:

1. **You are trained to miss them.** ~1 beat in 5 is shown to the network as a
   textbook example of "background", at exactly the record positions where
   reflection padding already makes the network weakest.
2. **The metric punishes you for finding them.** A perfect model that detects
   every beat scores *below* 1. Estimated ceiling with these labels
   (`2·|labelled| / (2·|labelled| + |true but unlabelled|)`):

   | class | labelled samples | true but unlabelled | Dice ceiling | baseline in the task text |
   |---|---|---|---|---|
   | P | 78 699 | ~23 240 | **0.87** | 0.74 |
   | QRS | 90 037 | ~19 505 | **0.90** | **0.89** |
   | T | 149 926 | ~37 765 | **0.89** | 0.85 |

   **The QRS baseline of 0.89 sits at the ceiling of 0.90.** Beating it on QRS
   is close to impossible while unlabelled beats count as false positives — no
   matter how good the network is. Fix the labels before you touch the model.

### Solutions

**A. Mask the unlabelled samples — recommended.** Give them target `-1` and skip
them in the loss and the metric. Part 1's `train()` already receives `loss_fn`,
so nothing shared has to change:

```python
y = np.zeros(len(x), np.int64)                     # between waves, a cardiologist
for c, onset, peak, offset in waves: y[onset:offset+1] = c   # looked and saw nothing
y[:waves[0][1]] = -1; y[waves[-1][3]+1:] = -1      # nobody looked here at all
loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
```
and in the metric, `m = target >= 0`, then Dice over `m` only. Keeps all 5000
samples per record, keeps the input shape, comparable with your classmates'
inputs. (This is the same trick Example 7 uses for padding — *"What must not
count is masked"*.)

**B. Crop to the annotated span.** Keep `[first_onset, last_offset]`. Nothing
partial is lost: the span begins at a wave onset and ends at a wave offset by
construction. Costs 27% of the samples, needs padding back to a fixed length or
per-record windows, and you must not then count the padding.

**C. Beat-anchored windows.** Take fixed windows (e.g. 1024 samples) centred on
each annotated QRS peak, from the span only. Kills the edge problem *and* the
class imbalance, gives small batches and a translation-invariant task; costs the
context between beats and needs a `Dataset` of windows rather than of records.

**D. Do not "fix" it by auto-labelling the edges** with your own detector. You
would train and evaluate against your own detector's errors, and the 0.74/0.89/0.85
reference numbers stop meaning anything.

---

## 2. The parser throws away 14 annotated QRS complexes

`if ann.symbol[k] in wave_code and ann.symbol[k-1] == "(" and ann.symbol[k+1] == ")"`
accepts only a perfect triplet. In three records the QRS is not written that way:

| record | stream as stored | effect |
|---|---|---|
| 95 | `N )` × 4 — no `(` | 4 QRS lost |
| 104 | `N )` × 9 — no `(`, the whole record | **all 9 QRS lost** |
| 111 | `) N (` at sample 3920 | 1 QRS lost |

Nothing is reported: the wave simply vanishes and its samples become background.
Record 104 ends up with labels that say *"no QRS anywhere in this record"* while
its annotation file describes nine of them (see `fig2_record104.png`). All three
records are in the **training** split, so the network is actively taught that a
QRS complex is background, 14 times.

*(Parenthesis counts give it away immediately: 4861 `(` vs 4874 `)` and 4874
peaks — the stream is not balanced.)*

### Solutions

* **Repair** — when a delimiter is missing, do not trust the one that survived
  either; centre a window of that class's median width (measured in the same
  record) on the peak. Implemented in `task1_data_cell_fixed.py`; recovers all
  14 waves with no overlaps, and every one of the 200 records then has a QRS.
* **Quarantine** — 3 records out of 200, all in training: simply drop them.
  Cheaper to justify than invented boundaries.
* **Whatever you do, assert afterwards.** The repaired cell ends by comparing the
  labelled QRS of each record with an independent beat detector inside the
  annotated span, counting only beats as tall as a labelled QRS. It prints:

  ```
  record | labelled QRS | beats with no label | rhythm
     138 |            8 |                  3 | Sinus rhythm
  too corrupted to check: [120, 126, 127, 133]
  records to look at: [95, 104, 111, 120, 126, 127, 133, 138]
  ```

  8 records of 200, each with a reason — versus 14 waves silently lost.

**Note the leftover:** record 138 has three real beats *inside* the annotated
span that carry no QRS label. Masking the edges (problem 1) does not cover that;
run the cross-check and mask or drop those beats too if you want a clean claim.

---

## 3. 24 records have no P wave at all

12% of records carry no P supervision (15 atrial fibrillation, 3 typical flutter,
paced rhythms, plus the malformed ones). In the **test** split, 4 of 30 records
have no P. The test set contains 223 P waves in 26 records, 288 QRS in 30, 261 T
in 30.

Consequence: per-record Dice for P is `0/0` on those records. Whether you skip
them, score them 1, or use a global (sum-over-records) Dice changes the reported
P number, and `#QRS != #P` in 186/200 records (mostly the unlabelled edges and
genuine beats without a P — PVC, blocked), so "count the waves" is not an option
either.

**Solution.** State the rule and use it consistently for model *and* baseline:
global Dice per class, or mean over records that contain the class, and report
`n` next to every number. With 26 P-bearing test records, differences below
~0.02 are noise — say so rather than reporting three decimals.

---

## 4. Class imbalance is real but overstated by problem 1

| | background | P | QRS | T |
|---|---|---|---|---|
| as the cell prints it | 68.1% | 7.9% | 9.0% | 15.0% |
| inside the annotated span | **56.4%** | **10.8%** | **12.4%** | **20.5%** |

A third of the "background" is missing labels (see `fig3_class_balance.png`).

**Solutions.** Masking (problem 1A) already improves the ratio. Beyond that:
`CrossEntropyLoss(weight=...)` set to inverse frequency, or a Dice/soft-Dice
term added to the loss. Note also the width spread — P 42–258 ms, QRS 56–252 ms,
T 70–320 ms, heart rate 30–167 bpm — so a single fixed receptive field is a
bottleneck: dilated convolutions or a deeper encoder with a wide coarse scale
help more than re-weighting.

---

## 5. Amplitude artefacts sit exactly where the labels are missing

13 records have their largest deflection of the whole record (|z| > 8) in the
**last 0.4 s**: 102, 109, 120, 121, 122, 124, 126, 127, 128, 129, 130, 132, 133.
Five records are also very low amplitude (std 0.035–0.05 mV against a median of
0.115 mV): 120, 126, 127, 130, 133. In 120, 126, 127 and 133 the combination is
bad enough that the calibration detector finds **no beat at all inside the
labelled span** — the record is nothing but a tail artefact and a flat line.

Because the cell standardizes with `(x - median) / std` over the *whole* record,
a tail artefact divides the amplitude of every real beat in that record.

**Solutions.** Compute the scaling statistics on the annotated span only (it is
the part you train on); or scale robustly — median and IQR / 5–95 percentile —
and clip to ±6; or band-pass 0.5–40 Hz before scaling; or drop these records.
Whichever you pick, report it, because it changes the input the network sees.

---

## 6. The released signals are already smoothed, and heavily quantized

Per 10 s record: only **300–600 distinct sample values**, 39–47% of adjacent
samples *exactly* equal, and exactly-flat runs with a median of 185 ms (up to
576 ms, record 128). Effectively no energy above ~40 Hz.

**Consequences.** There is no high-frequency detail to exploit — a "flat
segment" cue is nearly free, and the network will use it. Do not expect a model
trained here to behave the same on a raw clinical ECG. And eyeball the longest
flat runs: 0.5 s of a perfectly constant signal can also be a lead dropout.

---

## 7. Split: clean, but uneven in composition

Patient-clean (200 records from 200 subjects, zero overlap between train,
validation and test — verified). Rhythms parse exactly as the README states
(143 sinus, 25 bradycardia, 15 AF, 8 arrhythmia, 4 tachycardia, 3 flutter,
2 irregular). But:

| | train (140) | valid (30) | test (30) |
|---|---|---|---|
| no P wave | 18 | 2 | 4 |
| pacemaker | 7 | 0 | **3** |
| atrial fibrillation | 12 | 1 | 2 |
| ventricular extrasystole | 6 | 0 | 1 |

A third of the paced records land in the test set — paced QRS complexes are wide
and unusual, and there are only 3 of them. **Solution:** report the per-class
Dice with the rhythm breakdown, or add a per-rhythm row to your "What to show";
do not conclude your model is better on paced beats from 3 records.

---

## Checked and clean (so you don't spend time on these)

All records are exactly 5000 samples at 500 Hz; no NaN samples. Annotations are
time-ordered. No zero-width waves, no peak outside its `[onset, offset]`, no wave
overwrites another, no wave wider than 400 ms. The labels *are* aligned with the
waveform: with the local baseline removed, the annotated peak is within 20 ms of
the extremum of the wave in 94% of QRS, 98% of P and 97% of T waves. The rhythm
metadata parses correctly. The one thing the labels do not cover is the edges —
problem 1.

---

## Also: three things in your specification table (cell 47)

| your row | problem |
|---|---|
| network output `shape (3, 5000)` | Inconsistent with the task, which asks for **one of four classes per sample** → `(4, 5000)` logits with `CrossEntropyLoss`. `(3, 5000)` is only right if you switch to 3 independent sigmoid maps + `BCEWithLogitsLoss` and give up the background class. Pick one; the head and the loss must agree. |
| baseline `find peaks on the band passed signal` | That is step 1 of the task's baseline, not the baseline. The task defines: R peaks with `find_peaks` on the band-passed signal **plus** P/QRS/T placed at their median distances from the R peak measured on the *training* patients → 0.74 / 0.89 / 0.85. Implement that, or you have nothing to compare to. |
| last layer / loss / metric blank | Last layer `nn.Conv1d(c_out, 4, 1)` (1×1 conv, **no** activation — the softmax is in the loss). Loss `nn.CrossEntropyLoss(ignore_index=-1)`. Metric Dice per class over the unmasked samples, test set, next to the baseline's. |

`network input (1, 5000)` is correct.
