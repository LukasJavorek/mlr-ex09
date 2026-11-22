# Review — Agent-Built Tasks 1, 3, 6

Summary of specifications, verification checks, and agent errors found during audit.

---

## Task 1 — ECG Delineation (`task1_ludb/`)

* **Specification (`spec.py`):** Input `(1, 5000)`, output `(4, 5000)` (`P`, `QRS`, `T`, `background`), `CrossEntropyLoss(ignore_index=-1)`, margin 128, span-standardisation, class weights `0.5`, `UNet(1,1,4,[32,64,128,256,512])`, Adam `1e-3`, batch `16`, `75` epochs, detector `8–20 Hz / 200 ms / 0.35 / p90`. *Note: No `assert` binds the spec prose to the code.*
* **What was checked:**
  * **Smoke test (`results/smoke_log.txt`, 8 `[ok]` checks):** Cross-entropy matches a hand-computed mean over 14,594 annotated samples (within `1.3e-07`); Dice reaches `1.0` on label-built logits; overwriting all 5,406 unannotated predictions leaves the loss bit-identical (proving `ignore_index=-1` works).
  * **Baseline & Predictions:** Butterworth `8–20 Hz` + `find_peaks` baseline vs. UNet test Dice: **`0.738 / 0.881 / 0.829`** vs. **`0.758 / 0.908 / 0.822`** (`P / QRS / T`). Verified that `example_record.png` (record 121) was deterministically selected by `pick_record` (unique max of 15 QRS complexes).
* **What the agent got wrong:**
  * *Self-reported bugs:* Initial masking test modified targets instead of predictions; a `/2·FS` unit error caused absurd wave widths; a stage-A sweep compared a tuple to a list and returned all `nan`s.
  * *Found in audit:* `baseline.py` docstring claims `96` settings and `5–15 Hz / 240 ms`, whereas the code enumerates `120` settings and uses `8–20 Hz / 200 ms` (`choices.json`). `RESULTS.md` claims 7 `[ok]` lines (log has 8). `results/` is gitignored and the referenced `run_all_log.txt` does not exist. Most critically, `choices.json` contains a `note_widths` key written by no script—proving the file was hand-edited. Unresolved audit records (`120, 126, 127, 133, 138`) were omitted from "Honest limits".

---

## Task 3 — Spike Inference (`task3_spikefinder/`)

* **Specification (`spec.py`):** Input `(1, 512)` at 25 Hz (per-neuron z-scored) → 1 rate per 40 ms bin, `1×1` conv + softplus, Poisson NLL, binned Pearson correlation, window `128`, widths `[16,32,64,128]`, lr `3e-3`, batch `64`, `24` epochs, duplicate `(9,3)` dropped. *No prose-to-code `assert`.*
* **What was checked:**
  * **Smoke test (`smoke_test.py`, 9 checks / 28 asserts):** Verified Poisson NLL identity (`1e-6`), exact affine invariance of Pearson metric (`0.0e+00` deviation; constant input safely returns `None`), 30-window overfit reaching `0.994` correlation, and train/val/test split (`121 / 18 / 34` after dropping duplicate).
  * **Baseline & Predictions:** Test Pearson correlation: **`0.206`** (fluorescence level) and **`0.467`** (smoothed derivative) vs. **`0.632`** (model). `test_scores.png` plots all 34 held-out neurons individually.
* **What the agent got wrong:**
  * *Self-reported bugs:* `(9,3)` train/test duplicate leak (fixed after audit), `~0.026` optimism from selecting epochs on 18 validation neurons, and model collapse (`0.635 → 0.518`) at 90 epochs.
  * *Found in audit:* Log header prints `"0–6 spikes per 40 ms"` while `smoke_log.txt` two lines below observes `0–15`. `RESULTS.md` claims per-neuron standardisation adds **`+0.002`**, whereas `choices.json` shows `standardize none` (`0.6199`) beat standardised (`0.6176`)—the sign is flipped. The `±0.045` tie threshold is unsourced (audit gives `±0.022`, bootstrap gives `±0.063`).
  * *Timestamp contradiction:* File `mtime`s contradict the claimed workflow: `model.pt` (`13:13:53`) predates edits to `sf_data.py` (`13:14:11`) and `train_task3.py` (`13:14:24`); `eval_test_log.txt` (`13:22:35`) predates `eval_valid_log.txt` (`13:28:24`); and `smoke_log.txt` (`13:29:40`) was generated *after* training.

---

## Task 6 — Colourisation (`task6_colorize/`)

* **Specification (`spec.py`):** Enforces 4 key constants via `assert` (`16×16` grid, `256` bins, `±128` chroma units, `40 dB` clip)—the only task that programmatically prevents spec drift.
* **What was checked:**
  * **Smoke test (18 checks):** Reproduces all 5 pre-code data audit metrics within `0.02`; zero-initialised head matches the gray baseline to `0.00e+00` (confirming Y-channel pass-through and chroma scaling); subsampled `1024/4096` pixel loss matches full loss (`1.49e-07`); lookup ceilings verified (`31.54 dB`, `18.77% / 30.12%`).
  * **Baseline & Predictions:** Gray baseline (`26.94 / 25.36 / 25.58 dB` across splits). On the primary PSNR metric, gray (`25.36 dB`) beats the trained model (`24.26 dB`) because 10 near-B&W test images hit a `120 dB` floor. Seven test images plotted via rule ET5.
* **What the agent got wrong:**
  * *Disclosed & fixed:* Broken plotting calls (`figs.legend_on`, `figs.FIGURES`) caught by binding checks; cost tie-breaker favored fewer epochs via raw seconds; R4 override logged R4 while writing R3 hyperparameters; Run #1 headers leaked stale artifacts (resolved and verified via two identical 89-minute clean runs); minor reporting errors (quoting in-sample ceiling, misattributing validation α-cliff, `0.00458` vs `0.00464` prose typo).
  * *Still open:* `spec.text()` reads whatever JSONs exist in `results/`, so re-running without cleaning prints stale hyperparameters.

---

## Verdict

* **Task 6:** **Accepted.** Self-enforcing specification and two independent clean runs match bit-for-bit.
* **Task 1:** **Model accepted, record rejected.** Results are valid, but `choices.json` was hand-edited and the referenced master log does not exist.
* **Task 3:** **Model accepted with caveats.** `eval_valid_log.txt` reproduces `0.6349` from the shipped weights, but provenance is broken due to post-training code edits, out-of-order logs, and a sign-flipped claim.

## Reproduce

```bash
python smoke_test.py                     # Run in each task folder
python validate_choices.py --report      # Tasks 1 and 6
python spec.py                           # Task 6 (verifies the 4 spec asserts)
stat task3_spikefinder/results/*         # Verify Task 3 timestamp ordering


