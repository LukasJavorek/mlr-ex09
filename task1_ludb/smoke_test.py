"""Smoke test of the Task 1 pipeline. Run it before training anything:

    python smoke_test.py

Seven checks, each of which has printed output, and an assert that stops the run
if the answer is not what it should be:

  1  the data: shapes, dtypes, which targets occur, and that every record's
     annotated samples are one contiguous block - preceded by at most the context
     margin, followed by zeros carrying the target -1 (Option B)
  2  the shapes through the network: a batch in, the same length out, one channel
     per class
  3  the loss really skips the target -1 (compared against a hand-written mean of
     the per-sample cross-entropy over the annotated samples only)
  4  the class mapping: logits built from the labels score Dice 1.0 for P, QRS and
     T, so channel c of the output is the class the metric calls c
  5  what the network predicts where nobody annotated changes neither the loss
     nor the metric, so the score cannot be won or lost on the cropped edges
  6  the network can learn: a short run drives the loss down, and two records
     overfit to Dice > 0.95 in well under a minute
  7  the same batch twice gives the same answer (no dropout, eval mode)
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

import ludb_data
import metrics
import part1
from ludb_data import CLASSES, FS, IGNORE, LENGTH, WAVES, seed_everything
from part1 import UNet

WIDTHS = [32, 64, 128]


def ok(name: str, detail: str) -> None:
    print(f"[ok] {name}: {detail}")


def main() -> None:
    seed_everything(0)
    cfg = ludb_data.configured()
    data = ludb_data.build(cfg)                         # the same arrays the model will train on
    print(ludb_data.summary(data), "\n")

    # 1 -------------------------------------------------------------------- #
    X, Y = data["X"], data["Y"]
    assert X.shape == (200, LENGTH) and Y.shape == X.shape and X.dtype == np.float32 and Y.dtype == np.int64
    assert np.isfinite(X).all(), "a NaN in the input would poison the batch norm"
    assert set(np.unique(Y)) == {-1, 0, 1, 2, 3}, np.unique(Y)
    annotated = Y >= 0
    per_record = annotated.sum(1)
    start = annotated.argmax(1)                       # first annotated sample of each record
    pos = np.arange(LENGTH)[None, :]
    block = (pos >= start[:, None]) & (pos < (start + per_record)[:, None])
    assert (annotated == block).all(), "the annotated samples of a record must be one contiguous block"
    assert int(start.max()) <= cfg.margin, "only the context margin may sit in front of the span"
    padding = pos >= data["valid"][:, None]
    assert (Y[padding] == IGNORE).all() and (X[padding] == 0).all(), "the tail must be padding"
    std = np.array([X[i, s:s + n].std() for i, (s, n) in enumerate(zip(start, per_record))])
    assert np.allclose(std, 1, atol=0.02), std
    ok("data", f"X {X.shape} float32, Y {Y.shape} int64, targets {{-1,0,1,2,3}}, margin "
               f"{cfg.margin}: each record one contiguous block of {int(per_record.min())}-"
               f"{int(per_record.max())} annotated samples, starting at sample {int(start.min())}-"
               f"{int(start.max())} at the latest, standardized to std 1.00 on it, zeros and target "
               f"-1 behind it")

    # 2 -------------------------------------------------------------------- #
    sets = ludb_data.sets(data)
    batch_x, batch_y = sets["train"][0]
    assert batch_x.shape == (1, LENGTH) and batch_y.shape == (LENGTH,)
    model = UNet(dim=1, c_in=1, c_out=4, widths=WIDTHS).to(part1.device)
    x = torch.stack([sets["train"][i][0] for i in range(4)])
    y = torch.stack([sets["train"][i][1] for i in range(4)])
    assert x.shape == (4, 1, LENGTH) and y.shape == (4, LENGTH)
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 4, LENGTH), out.shape
    pred = out.argmax(1)
    assert pred.shape == (4, LENGTH) and set(np.unique(pred)) <= set(range(4))
    ok("shapes", f"input {tuple(x.shape)} -> output {tuple(out.shape)} "
                 f"(4 logits per sample, no activation), argmax {tuple(pred.shape)} with values "
                 f"{sorted(set(int(v) for v in pred.unique()))} = {[WAVES[c] for c in sorted(set(int(v) for v in pred.unique()))]}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"     UNet(dim=1, c_in=1, c_out=4, widths={WIDTHS}): {n_params / 1e6:.2f} M parameters")

    # 3 -------------------------------------------------------------------- #
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE)
    with torch.no_grad():
        logits = model(x).double()
        by_loss = loss_fn(logits.float(), y).item()
        log_prob = torch.log_softmax(logits, dim=1)
        keep = y >= 0
        by_hand = -log_prob.gather(1, y.clamp(min=0)[:, None]).squeeze(1)[keep].mean().item()
    assert abs(by_loss - by_hand) < 1e-4, (by_loss, by_hand)
    ok("loss", f"CrossEntropyLoss(ignore_index=-1) = {by_loss:.5f} equals the hand-written mean over "
               f"the {int(keep.sum())} annotated samples, {by_loss - by_hand:+.2e}")

    # 4 -------------------------------------------------------------------- #
    cheat = torch.full_like(out, -30.0)
    cheat.scatter_(1, y.clamp(min=0)[:, None, :], 30.0)         # one-hot logits of the labels
    with torch.no_grad():
        cheat_loss = nn.CrossEntropyLoss(ignore_index=IGNORE)(cheat, y).item()
    scores = metrics.evaluate(cheat.argmax(1).numpy(), y.numpy())
    assert cheat_loss < 1e-6, cheat_loss
    for c in CLASSES:
        assert scores[WAVES[c]]["global"] == 1.0, scores
    ok("class mapping", f"logits made from the labels give loss {cheat_loss:.1e} and Dice "
                        f"{[round(scores[WAVES[c]]['global'], 3) for c in CLASSES]} for "
                        f"{[WAVES[c] for c in CLASSES]} -> channel c is class c, in the loss and in the metric")

    # 5 -------------------------------------------------------------------- #
    keep = y >= 0
    mask = (~keep)[:, None, :]
    # at every unannotated sample, replace whatever the logits said by a loud "background"
    tampered = logits.detach().masked_fill(mask, 0.0)
    tampered[:, 0] = tampered[:, 0].masked_fill(~keep, 50.0)
    perfect = cheat.detach().masked_fill(mask, 0.0)          # the same tampering on the
    perfect[:, 0] = perfect[:, 0].masked_fill(~keep, 50.0)   # one-hot logits of check 4
    with torch.no_grad():
        same_loss = loss_fn(logits.float(), y).item() == loss_fn(tampered.float(), y).item()
        before = metrics.evaluate(cheat.argmax(1).numpy(), y.numpy())["QRS"]["global"]
        after = metrics.evaluate(perfect.argmax(1).numpy(), y.numpy())["QRS"]["global"]
    assert same_loss and before == after == 1.0, (same_loss, before, after)
    ok("masking", f"overwrote the prediction at all {int((~keep).sum())} unannotated samples of the "
                  f"batch: the real logits still give the same loss {by_loss:.5f} to the bit, and a "
                  f"segmentation worth Dice {before:.1f} is still worth {after:.1f} - what the network "
                  f"says outside the span cannot win or lose the score")

    # 6 -------------------------------------------------------------------- #
    small = torch.utils.data.Subset(sets["train"], list(range(8)))
    loader = torch.utils.data.DataLoader(small, batch_size=4, shuffle=True)
    model = UNet(dim=1, c_in=1, c_out=4, widths=WIDTHS).to(part1.device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    curve = []
    for step in range(40):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = nn.CrossEntropyLoss(ignore_index=IGNORE)(model(xb.to(part1.device)), yb.to(part1.device))
            loss.backward()
            opt.step()
            curve.append(loss.item())
    first, last = np.mean(curve[:5]), np.mean(curve[-5:])
    assert last < 0.6 * first, (first, last)
    ok("learning", f"80 batches of 4 records: loss {first:.3f} -> {last:.3f} "
                   f"({last / first:.0%} of the start)")

    two = torch.utils.data.Subset(sets["train"], list(range(2)))
    model = UNet(dim=1, c_in=1, c_out=4, widths=WIDTHS).to(part1.device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    x2 = torch.stack([two[i][0] for i in range(2)])
    y2 = torch.stack([two[i][1] for i in range(2)])
    for step in range(300):
        model.train()
        opt.zero_grad()
        loss = nn.CrossEntropyLoss(ignore_index=IGNORE)(model(x2.to(part1.device)), y2.to(part1.device))
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        overfit = model(x2.to(part1.device)).argmax(1).cpu().numpy()
    overfit_scores = metrics.evaluate(overfit, y2.numpy())
    assert all(overfit_scores[WAVES[c]]["global"] > 0.95 for c in CLASSES), overfit_scores
    ok("capacity", f"2 records, 300 steps: loss {loss.item():.4f}, Dice "
                   f"{[round(overfit_scores[WAVES[c]]['global'], 3) for c in CLASSES]} "
                   f"- the network can represent these labels")

    # 7 -------------------------------------------------------------------- #
    model.eval()
    with torch.no_grad():
        a, b = model(x2.to(part1.device)), model(x2.to(part1.device))
    assert torch.equal(a, b)
    ok("determinism", "the same batch twice gives bit-identical outputs in eval mode")

    # the timing of the real training, measured on its own batch size
    n = 16
    xb = torch.stack([sets["train"][i][0] for i in range(n)]).to(part1.device)
    yb = torch.stack([sets["train"][i][1] for i in range(n)]).to(part1.device)
    model.train()
    t0 = time.perf_counter()
    for _ in range(3):
        opt.zero_grad()
        nn.CrossEntropyLoss(ignore_index=IGNORE)(model(xb), yb).backward()
        opt.step()
    per_batch = (time.perf_counter() - t0) / 3
    epochs_guess = per_batch * -(-len(sets["train"]) // n)          # ceil: batches per epoch
    print(f"\nsmoke test passed.")
    print(f"timing: {per_batch:.2f} s per batch of {n} on {part1.device}, so "
          f"~{epochs_guess:.0f} s per epoch over the {len(sets['train'])} training records, "
          f"~{2 * epochs_guess * 30 / 60:.0f} min for 30 epochs including validation.")


if __name__ == "__main__":
    main()
