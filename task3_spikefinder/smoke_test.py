"""Smoke test of the Task 3 pipeline. Run it before training anything:

    python smoke_test.py

Nine checks, each with printed output and an assert that stops the run when the
answer is not what it should be:

  1  the data: 174 neurons, the shapes and dtypes, the counts are counts, the
     standardization is per neuron, and no test neuron carries a target scale
  2  no window contains padding - checked against the csv files again through a
     second code path (`baseline.traces`), not against the loader's own bookkeeping
  3  the split is the notebook's: per-dataset sizes, 122/18/34, and the dataset 9
     duplicate really does straddle train and test - which is what the drop fixes
  4  shapes through the network: a batch in, the same length out, one channel, and a
     whole recording of an odd length comes out the same odd length - the claim
     `fitting.predict_trace` rests on
  5  the loss is the Poisson likelihood of the *real* counts: for one window,
     mean(rho - r log rho) equals mean(lambda - y log lambda)/mu + mean(r) log mu to
     1e-4, i.e. it differs from the textbook NLL only by a constant per neuron that
     cannot move a gradient
  6  the metric: a perfect prediction scores 1.0, any positive affine map of a
     prediction scores *identically* (the whole relative-rate design stands on this),
     a constant prediction is undefined rather than 0, and predictions scored against
     somebody else's spike train score about zero
  7  the network can learn this mapping: a handful of windows overfit, and their
     correlation goes from chance to high - scored over the windows that contain a
     spike, since a window whose target is empty throughout has no correlation to
     score, which is the metric's own None rule (check 6) applied here too
  8  determinism: the same batch twice gives bit-identical outputs in eval mode
  9  the head really is non-negative, including where the linear layer says loudly
     negative
"""
from __future__ import annotations

import time

import numpy as np
import torch

import baseline
import fitting
import metrics
import part1
import sf_data
from sf_data import BIN, FS_IN, TWIN, seed_everything

WIDTHS = [16, 32, 64]


def ok(name: str, detail: str) -> None:
    print(f"[ok] {name}: {detail}")


def main() -> None:
    seed_everything(0)
    cfg = sf_data.configured()
    data = sf_data.build(cfg)                          # the same arrays the model trains on
    # train and valid windows only: the split checks below compare neuron *keys* of all
    # three splits, but no check here computes anything from a test window, so none is built
    sets = sf_data.sets(data, ("train", "valid"))
    print()

    # 1 -------------------------------------------------------------------- #
    x, y = data["x"], data["y"]
    assert len(data["neurons"]) == 174 and set(x) == set(y) == set(data["neurons"])
    assert all(v.dtype == np.float32 for v in x.values())
    assert all(np.isfinite(v).all() for v in x.values()), "a NaN would poison the batch norm"
    counts = np.concatenate([v for v in y.values()])
    assert np.allclose(counts, np.round(counts)) and counts.min() == 0 and counts.max() <= 6 * BIN
    off = [n for n, v in x.items() if abs(v.mean()) > 1e-3 or abs(v.std() - 1) > 2e-2]
    assert not off, f"standardization is not per neuron for {off[:5]}"
    assert data["mu"].keys() == set(data["split"]["train"] + data["split"]["valid"]), \
        "a test neuron must not have a target scale: the metric cannot see one"
    ok("data", f"174 neurons; x standardized per neuron (worst |mean| "
               f"{max(abs(v.mean()) for v in x.values()):.1e}, worst |std-1| "
               f"{max(abs(v.std() - 1) for v in x.values()):.1e}); y integer counts per 40 ms bin, "
               f"0-{int(counts.max())} observed ({6 * BIN} is the ceiling: {6} spikes in {BIN} "
               f"samples), {(counts == 0).mean():.1%} empty, mean {counts.mean():.3f} per bin; "
               f"target scales exist for the {len(data['mu'])} train+valid neurons and for no "
               f"test neuron")

    # 2 -------------------------------------------------------------------- #
    raw = baseline.traces()                            # the csv read again, by another path
    too_long = [n for n, s in data["starts"].items()
                if len(s) and (s[-1] + cfg.window) * cfg.decimate > len(raw[n])]
    assert not too_long, f"windows reach past the real signal for {too_long[:5]}"
    real = sum(len(v) for v in raw.values()) / FS_IN          # seconds, at the released 100 Hz
    covered = sum(len(s) for s in data["starts"].values()) * cfg.seconds
    assert all(len(s) >= 2 for s in data["starts"].values()), \
        "a neuron shorter than two windows: the last one would be half padding"
    ok("no padding", f"every one of the {int(covered / cfg.seconds)} windows of {cfg.window} samples "
                     f"ends at or before its neuron's last real sample, checked against {len(raw)} raw "
                     f"columns re-read from the csv; {real - covered:.0f} s of the {real:.0f} s "
                     f"recorded - the ragged tail under one window per neuron - is all the signal no "
                     f"window sees, and no window is padded to reach {cfg.window}")

    # 3 -------------------------------------------------------------------- #
    sizes = {k: len(v) for k, v in data["split"].items()}
    notebook = sf_data.split_of(data["neurons"], cfg.seed)
    assert {k: len(v) for k, v in notebook.items()} == {"train": 122, "valid": 18, "test": 34}
    assert sizes == {"train": 121 if cfg.drop_duplicate else 122, "valid": 18, "test": 34}, sizes
    for ds in sf_data.DATASETS:
        k = len([nn for nn in data["neurons"] if nn[0] == ds])
        assert len([nn for nn in notebook["test"] if nn[0] == ds]) == max(1, round(0.2 * k))
    assert TWIN not in data["split"]["train"] and (9, 4) in data["split"]["test"]
    assert not set(data["split"]["train"]) & (set(data["split"]["valid"]) | set(data["split"]["test"]))
    ok("split", f"the notebook's rng(0) split rebuilt to the neuron: 122/18/34 with per-dataset "
                f"n_test = max(1, round(0.2n)); dataset 9 really did hold the leak - (9,3) trained on "
                f"while its duplicate (9,4) sits in test - and with (9,3) dropped the "
                f"{sizes['train']} training neurons share no recording with any of the "
                f"{sizes['test']} test neurons")

    # 4 -------------------------------------------------------------------- #
    model = fitting.RateNet(WIDTHS).to(part1.device)
    xb = torch.stack([sets["train"][i][0] for i in range(4)])
    assert xb.shape == (4, 1, cfg.window), xb.shape
    model.eval()
    with torch.no_grad():
        out = model(xb)
    assert out.shape == (4, 1, cfg.window) and torch.isfinite(out).all()
    odd = next(len(data["x"][k]) for k in data["split"]["valid"] if len(data["x"][k]) % 8)
    with torch.no_grad():
        long_out = model(torch.zeros(1, 1, odd)).shape
    assert long_out == (1, 1, odd), (odd, long_out)
    ok("shapes", f"input {tuple(xb.shape)} -> output {tuple(out.shape)}: one non-negative rate per "
                 f"40 ms bin, nothing cropped; a whole recording of {odd} samples (not a multiple of "
                 f"8) comes out {odd} samples too, so a test neuron is one forward pass and no "
                 f"stitching of overlapping windows")

    # 5 -------------------------------------------------------------------- #
    one = sets["train"][0]
    rel = one[1][None].double()                        # (1, 1, window), counts / mu
    mu0 = float(data["mu"][data["split"]["train"][0]])
    rho = torch.rand_like(rel) * 3 + 0.05              # any positive rates will do
    counts_b = rel * mu0                               # the real counts per bin
    lam = rho * mu0
    as_relative = fitting.poisson_nll(rho, rel).item()
    as_counts = ((lam - counts_b * torch.log(lam)).mean() / mu0 + rel.mean() * np.log(mu0)).item()
    assert abs(as_relative - as_counts) < 1e-4, (as_relative, as_counts)
    assert fitting.poisson_nll(torch.full_like(rel, 1.0), rel) \
        < fitting.poisson_nll(torch.full_like(rel, 8.0), rel)
    ok("loss", f"poisson_nll(rho, r) = {as_relative:.5f} equals the textbook mean(lambda - y log "
               f"lambda) of the real counts divided by that neuron's mu plus mean(r) log mu "
               f"({as_relative - as_counts:+.1e}): one likelihood, renormalized per neuron, so the "
               f"12 spikes/s neurons cannot outweigh the 0.02 spikes/s ones")

    # 6 -------------------------------------------------------------------- #
    key = data["split"]["valid"][0]
    truth = data["y"][key]
    maps = {"x2": truth * 2.0, "x2 as float32": (truth * 2.0).astype(np.float32),
            "x1e-6": truth * 1e-6, f"/mu ({data['mu'][key]:.3f})": truth / data["mu"][key],
            "-mean": truth - truth.mean(), "+1e6": truth + 1e6}
    affine = {k: metrics.correlate(v, truth) for k, v in maps.items()}
    worst = max(abs(v - 1.0) for v in affine.values())
    assert metrics.correlate(truth, truth) == 1.0 and worst < 1e-9, (affine, worst)
    assert metrics.correlate(np.ones_like(truth), truth) is None
    preds = fitting.predict_split(model, data, "valid")
    controls = metrics.control_means(preds, {k: data["y"][k] for k in data["split"]["valid"]})
    assert abs(controls["shifted"]) < 0.05, controls
    ok("metric", f"a perfect prediction scores 1.0 and so does every positive affine map of it "
                 f"({', '.join(maps)}), the worst deviating from 1.0 by {worst:.1e} - the scale "
                 f"the network does not know is a scale the metric cannot see; a constant "
                 f"prediction is undefined (None), not 0, so a flat line is caught rather than "
                 f"averaged away; and untrained predictions rolled through time score "
                 f"{controls['shifted']:+.3f} (against another neuron: {controls['shuffled']:+.3f}, "
                 f"above zero only because same-session neurons share signal) - the score "
                 f"measures alignment")

    # 7 -------------------------------------------------------------------- #
    idx = list(range(32))
    small = torch.utils.data.Subset(sets["train"], idx)
    target = np.stack([sets["train"][i][1][0].numpy() for i in idx])   # (windows, bins)
    model = fitting.RateNet(WIDTHS).to(part1.device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    curve = []
    for _ in range(60):
        model.train()
        for xs, ys in torch.utils.data.DataLoader(small, batch_size=16, shuffle=True):
            opt.zero_grad()
            loss = fitting.poisson_nll(model(xs.to(part1.device)), ys.to(part1.device))
            loss.backward()
            opt.step()
            curve.append(loss.item())
    first, last = np.mean(curve[:5]), np.mean(curve[-5:])
    assert last < 0.75 * first, (first, last)
    model.eval()
    with torch.no_grad():
        fitted = model(torch.stack([small[i][0] for i in idx]).to(part1.device)).cpu().numpy()[:, 0]
    per = [metrics.correlate(fitted[i], target[i]) for i in idx]
    got = [v for v in per if v is not None]
    silent = len(idx) - len(got)
    assert all(target[i].sum() == 0 for i, v in zip(idx, per) if v is None), \
        "a window scored None that does contain spikes: the network output, not the target, is flat"
    assert len(got) > len(idx) // 2, f"only {len(got)} of {len(idx)} windows are scorable"
    overfit = float(np.mean(got))
    assert overfit > 0.5, overfit
    empty = np.mean([data["y"][k][a:a + cfg.window].sum() == 0
                     for k in data["split"]["train"] for a in data["starts"][k]])
    ok("capacity", f"32 windows, 120 batches: loss {first:.3f} -> {last:.3f} and the correlation on "
                   f"those same windows {overfit:.3f}, measured over the {len(got)} that carry a "
                   f"spike - the {silent} that are not scorable are windows whose target is empty "
                   f"throughout, where the metric is undefined by its own rule (check 6) rather than "
                   f"zero, as it is for {empty:.0%} of all the training windows at this window length "
                   f"- the network can represent this mapping, so a low score later would be the data "
                   f"or the settings, not the architecture")

    # 8 -------------------------------------------------------------------- #
    with torch.no_grad():
        a, b = model(xb.to(part1.device)), model(xb.to(part1.device))
    assert torch.equal(a, b)
    ok("determinism", "the same batch twice gives bit-identical outputs in eval mode")

    # 9 -------------------------------------------------------------------- #
    loud = fitting.RateNet(WIDTHS)
    with torch.no_grad():
        loud.net.head.weight.mul_(400.0)               # make the 1 x 1 head shout
        head_out = loud.net(xb)                        # its output, before softplus
        neg = loud(xb)                                 # and after
    assert (neg >= 0).all() and torch.isfinite(neg).all() and (neg < 1e-3).any()
    ok("head", f"with the head driven to logits down to {head_out.min():.0f} the output stays in "
               f"[{neg.min():.2e}, {neg.max():.2f}] and finite: softplus is non-negative everywhere "
               f"and still has a gradient at 0, where {(counts == 0).mean():.0%} of the bins sit")

    # the timing of the real training, measured on its own batch size, after a warm-up:
    # the first few batches of a process pay for kernel selection and are ~8x slower
    n, warm = 32, 3
    xb = torch.stack([sets["train"][i][0] for i in range(n)]).to(part1.device)
    yb = torch.stack([sets["train"][i][1] for i in range(n)]).to(part1.device)
    model = fitting.RateNet(WIDTHS).to(part1.device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(warm):
        opt.zero_grad()
        fitting.poisson_nll(model(xb), yb).backward()
        opt.step()
    t0 = time.perf_counter()
    for _ in range(5):
        opt.zero_grad()
        fitting.poisson_nll(model(xb), yb).backward()
        opt.step()
    per_batch = (time.perf_counter() - t0) / 5
    steps = -(-len(sets["train"]) // n)
    print("\nsmoke test passed.")
    print(f"timing: {per_batch * 1000:.0f} ms per batch of {n} on {part1.device} with "
          f"{torch.get_num_threads()} torch threads (warm), so ~{per_batch * steps:.1f} s per "
          f"epoch over the {len(sets['train'])} training windows ({steps} steps) at "
          f"widths={WIDTHS}; the sweeps and the final run use more levels and set 6 threads, "
          f"and their real seconds are in results/choices.json and results/train_log.json.")


if __name__ == "__main__":
    main()
