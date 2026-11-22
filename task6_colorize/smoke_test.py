"""Everything to run before anything is trained: 18 printed checks, in 17 numbered blocks.
`python smoke_test.py`, ~30 s.

The 18 is not remembered, it is counted: `ok()` increments a counter and the last line of
`main()` asserts it equals `CHECKS` below, so the number the other scripts quote from this one
cannot quietly fall behind the code. (One block prints two lines, for the two controls.)

Check 17 is the one that draws pictures: it runs the three figures of `evaluate_task6.py` on
eight synthetic photos, into a temporary directory that it deletes again, so this script still
leaves nothing behind.

Every one of these is a claim the specification makes about the code, turned into an assert
with a printed number next to it. They exist because the way a colourisation pipeline goes
wrong is invisible in its loss: a chroma channel swapped for the other one, a +-128 scale
applied twice, Y predicted instead of copied, a bin grid one cell off from the one the
ceilings were computed on - all train, all fall in loss, all produce washed-out or
neon pictures that nobody can trace to a line. Several of them are checks against numbers
measured in `../task6_data_audit.md` *before this code existed*, which is the only kind of
check that can catch a wrong-but-plausible implementation.

Nothing here trains a network. The two models it builds are untrained on purpose: the
zero-initialised route-A head must score *exactly* the gray baseline, which is the free
proof that Y is copied back and that the output scale is right, and it stops being free the
moment a model has been fitted.
"""
from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import colorspace as cs
import col6_data as d6
import evaluate_task6 as E        # check 17 runs its three figure functions on fake arrays
import figs                       # check 17 redirects figs.FIGURES at a temporary directory
import fitting
import metrics as M

# the folder's own helper modules: check 16 binds every call site in every script to these
MODS = {m: importlib.import_module(m) for m in
        ("colorspace", "col6_data", "figs", "fitting", "metrics", "spec", "evaluate_task6")}


CHECKS = 18             # what main() actually prints, asserted at its end - do not edit freely
_printed = 0


def ok(name: str, detail: str) -> None:
    global _printed
    _printed += 1
    print(f"[ok] {name}: {detail}")


def main() -> None:
    global _printed
    _printed = 0
    sys.stdout.reconfigure(line_buffering=True)
    fitting.threads(6)
    d6.seed_everything(0)
    t0 = time.perf_counter()
    data = d6.build(verbose=False)
    train, valid = d6.arrays(data, "train"), d6.arrays(data, "valid")
    test = d6.arrays(data, "test")

    # 1 -------------------------------------------------------------------------------------- #
    assert train.shape == (9469, 64, 64, 3) and valid.shape == (100, 64, 64, 3) \
        and test.shape == (500, 64, 64, 3), [v.shape for v in (train, valid, test)]
    assert data["split"]["valrest"] and len(data["split"]["valrest"]) == 3325
    ok("data", f"train {train.shape}, valid {valid.shape}, test {test.shape}, "
               f"valrest {len(data['split']['valrest'])} - the notebook's split sizes")

    # 2 -------------------------------------------------------------------------------------- #
    # The pixels are the audit's pixels, and the split is the notebook's split.
    lines = d6.audit_crosscheck(data)
    assert all("identical" in line for line in lines), lines
    val_union = sorted(data["split"]["valid"] + data["split"]["test"] + data["split"]["valrest"])
    assert val_union == sorted(data["files"]["val"]), "the three val splits must partition val/"
    assert len(set(data["split"]["valid"]) & set(data["split"]["test"])) == 0, "valid/test overlap"
    ok("split and cache", f"all three arrays byte-identical to the audit's; "
                          f"valid + test + valrest partition the 3925 val photos exactly; "
                          f"{' and '.join(l.split('sha1')[1].strip() for l in lines)}")

    # 3 -------------------------------------------------------------------------------------- #
    # The audit's data facts, re-derived here rather than trusted.
    cf = {s: cs.colourfulness(*cs.ycc(d6.arrays(data, s))[1:]) for s in
          ("train", "valid", "test")}
    for split, want in (("train", 0.0424), ("valid", 0.0402), ("test", 0.0407)):
        assert abs(cf[split] - want) < 1e-3, (split, cf[split], want)
    grey = {s: int(d6.exactly_grey(d6.arrays(data, s)).sum())
            for s in ("train", "valid", "test")}
    assert grey == {"train": 181, "valid": 3, "test": 8}, grey
    ok("the audit's data facts", f"colourfulness train/valid/test "
                                 f"{cf['train']:.4f} / {cf['valid']:.4f} / {cf['test']:.4f} "
                                 f"(the audit: 0.0424 / 0.0402 / 0.0407); photos already black "
                                 f"and white {grey} (the audit: 181 / 3 / 8)")

    # 4 -------------------------------------------------------------------------------------- #
    resid = cs.roundtrip_worst(valid)
    assert resid < 5e-3, resid
    y, cb, cr = cs.ycc(valid)
    gray = M.baseline_photos(valid)
    gray_chroma = cs.colourfulness(*cs.ycc(gray)[1:])
    assert gray_chroma < 1e-4, gray_chroma           # float32 residue, quantified below
    assert np.abs(cs.ycc(gray)[0] - y).max() < 1e-3, "rgb_of(Y, 0, 0) must preserve Y"
    # A grey stack of Y written directly is the other way to say "no colour"; the two must
    # score identically, or the baseline depends on a choice nobody made.
    flat = np.stack([y] * 3, -1)
    gap = float(np.abs(M.psnr_photos(gray, valid) - M.psnr_photos(flat, valid)).max())
    assert gap < 1e-9, gap
    ok("colour transform", f"round-trip residue {resid:.5f} of 255 on 100 real photos; the gray "
                           f"baseline's chroma is {gray_chroma:.1e} - not exactly 0, because "
                           f"0.299+0.587+0.114 is not exactly 1 in float32, so recomputing Y "
                           f"from R=G=B=Y leaves a residue at float precision, seven orders of "
                           f"magnitude below the 0.0402 this measures on real colour; its Y differs from "
                           f"the input Y by at most {np.abs(cs.ycc(gray)[0] - y).max():.2e}; and "
                           f"grey written directly as a stack of Y scores the same PSNR to "
                           f"{gap:.1e} dB, so the baseline does not depend on the trip "
                           f"through rgb_of")

    # 5 -------------------------------------------------------------------------------------- #
    # THE check that the metric is the audit's metric: the gray baseline, re-scored here, must
    # land on the numbers in the specification table (audit section 2, valid column).
    s_gray = M.score(gray, valid, baseline_per_photo=M.baseline_per_photo(valid))
    want = {"psnr_mean": 26.94, "psnr_clip40": 24.46, "psnr_pooled": 21.06,
            "psnr_median": 23.95, "chroma_l1": 0.0804}
    drift = {k: (s_gray[k], v) for k, v in want.items() if abs(s_gray[k] - v) > 0.02}
    assert not drift, f"the metric does not reproduce the audit's baseline: {drift}"
    assert abs(s_gray["chroma_l1"] - 2 * s_gray["colourfulness_truth"]) < 2e-4
    assert s_gray["beats_baseline"] == 0, "nothing beats itself"
    ok("metric == the audit's metric",
       f"gray on valid: {s_gray['psnr_mean']:.2f} dB mean (audit 26.94), "
       f"{s_gray['psnr_clip40']:.2f} clipped at 40 (24.46), {s_gray['psnr_pooled']:.2f} pooled "
       f"(21.06), median {s_gray['psnr_median']:.2f} (23.95), chroma L1 {s_gray['chroma_l1']:.4f} "
       f"(0.0804) - all within 0.02; chromaL1 == 2 x truth colourfulness "
       f"({s_gray['colourfulness_truth']:.4f}) as the two definitions require")

    # 6 -------------------------------------------------------------------------------------- #
    # A zero-initialised route-A head must be the gray baseline, to the digit.
    model_a = fitting.Colorizer("A", [16, 32, 64])
    out = fitting.decode(model_a, valid[:20])
    per_model = M.psnr_photos(out["rgb"], valid[:20])
    per_gray = M.psnr_photos(M.baseline_photos(valid[:20]), valid[:20])
    assert np.abs(out["cb"]).max() == 0.0 and np.abs(out["cr"]).max() == 0.0
    gap = float(np.abs(per_model - per_gray).max())
    assert gap < 1e-6, gap
    ok("zero-init route A == the baseline", f"max per-photo PSNR difference {gap:.2e} dB over "
                                            f"20 photos, and its chroma is exactly 0 - so Y is "
                                            f"written back unchanged and the +-128 scale is "
                                            f"applied once, not twice")

    # 7 -------------------------------------------------------------------------------------- #
    # Route B's init is "the training prior, nothing else". Done on a FRESH model, before the
    # head is randomised below - the point of the check is the initialisation itself.
    prior = fitting.prior(train, cs.GRID)
    model_b = fitting.Colorizer("B", [16, 32, 64], prior_prob=prior)
    x = torch.from_numpy(d6.inputs_of(valid[:4])).to(fitting.part1.device)   # (4, 1, 64, 64)
    with torch.no_grad():
        logits = model_b.unet.head(model_b.features(x)).permute(0, 2, 3, 1).double()
    # The stable softmax written out the way `fitting.decode` does it, against torch's own:
    # the one place a hand-rolled log-softmax could hide an error is the inference path.
    m = logits.max(-1, keepdim=True).values
    logp = (logits - m - (logits - m).exp().sum(-1, keepdim=True).log()).cpu().numpy()
    ref = torch.log_softmax(logits, -1).cpu().numpy()
    assert np.abs(logp - ref).max() < 1e-9, np.abs(logp - ref).max()
    p = np.exp(logp).reshape(-1, cs.GRID ** 2)
    spread = float(np.abs(p - p[0]).max())
    d_prior = float(np.abs(p[0] - prior).max())
    ce_at_init = float(-(p * np.log(prior)[None, :]).sum(-1).mean())
    h_prior = float(-(prior * np.log(prior)).sum())
    assert spread < 1e-5, spread
    assert d_prior < 1e-4, d_prior
    assert abs(ce_at_init - h_prior) < 1e-4, (ce_at_init, h_prior)
    ok("route-B init == the prior", f"p is constant across pixels and photos (spread "
                                    f"{spread:.1e}) and equals the training marginal to "
                                    f"{d_prior:.1e}; its cross-entropy is {ce_at_init:.4f} nats "
                                    f"= H(prior) {h_prior:.4f} (not log 256 = "
                                    f"{np.log(256):.3f}, which a zero-bias init would give). "
                                    f"An epoch-1 route-B model therefore starts at the prior's "
                                    f"own accuracy, so the sweep's first epoch is readable; the "
                                    f"max-subtracting log-softmax that `decode` writes by hand "
                                    f"agrees with `torch.log_softmax` to "
                                    f"{np.abs(logp - ref).max():.1e}")

    # 8 -------------------------------------------------------------------------------------- #
    # The pixel-indexed head must be the same function as the full head. The head is
    # randomised first: at the zeroed init both paths return the bias and agree for free.
    torch.manual_seed(0)
    with torch.no_grad():
        model_b.unet.head.weight.normal_(0, 0.05)
        model_b.unet.head.bias.normal_(0, 0.05)
    with torch.no_grad():
        feats = model_b.features(x)
        full = model_b.head(feats).permute(0, 2, 3, 1)
        full = full.reshape(-1, full.shape[-1])
        idx = torch.arange(0, x.shape[0] * 64 * 64, 7, device=feats.device)
        sampled = model_b.head(feats, idx)
    assert full.shape[1] == cs.GRID ** 2 and sampled.shape == (len(idx), cs.GRID ** 2)
    diff = float((sampled - full[idx.cpu()]).abs().max())
    assert diff < 1e-4, diff
    ok("sampled head == full head", f"max |difference| {diff:.2e} over {len(idx)} pixels of a "
                                    f"randomly-initialised route-B model ({sampled.shape[1]} "
                                    f"logits): the 1024-of-4096 pixel loss is the same "
                                    f"objective, not a different one")

    # 9 -------------------------------------------------------------------------------------- #
    # The bin grid is the grid the audit's ceilings were computed on.
    idx_check = cs.bin_index(cb, cr, cs.GRID)
    cb2, cr2 = cs.bin_centre(idx_check, cs.GRID)
    step = 2 * cs.CHROMA_SCALE / cs.GRID
    assert np.abs(cb - cb2).max() <= step / 2 + 1e-4 and np.abs(cr - cr2).max() <= step / 2 + 1e-4
    binned = M.binned_truth(valid, cs.GRID)
    s_bin = M.score(binned, valid)
    assert abs(s_bin["psnr_mean"] - 31.54) < 0.05, s_bin["psnr_mean"]
    ok("the bin grid", f"every cell centre within {step / 2:.1f} of its pixels; the 256-bin "
                       f"binned truth scores {s_bin['psnr_mean']:.2f} dB on valid, the audit's "
                       f"route-B ceiling (31.54) - so this is the grid those ceilings are for")

    # 10 ------------------------------------------------------------------------------------- #
    # The ceilings, recomputed. The 32x32 arm must reproduce the audit's published number.
    ceil32 = M.ceilings(train, 32)
    ceil16 = M.ceilings(train, cs.GRID)
    assert abs(ceil32["accuracy_pct"] - 18.8) < 0.5, ceil32
    assert ceil16["accuracy_pct"] > ceil32["accuracy_pct"], "a coarser grid must not be harder"
    assert ceil16["ce_floor_nats"] < ceil32["ce_floor_nats"]
    ok("ceilings, and the grid they depend on",
       f"32x32: lookup-table accuracy {ceil32['accuracy_pct']:.2f}% (the audit's 18.8%), "
       f"H(C|Y) {ceil32['ce_floor_nats']} nats against H(C) {ceil32['entropy_nats']}; the 16x16 "
       f"grid this folder actually trains has {ceil16['accuracy_pct']:.2f}% and "
       f"{ceil16['ce_floor_nats']} nats against H(C) {ceil16['entropy_nats']} "
       f"({ceil16['cells_populated']} of {ceil16['cells_total']} cells populated, top cell "
       f"{ceil16['top_cell_share_pct']}% of the pixels). Two things this pins down. The 18.8% "
       f"the audit published is the 32x32 grid it analysed, so route B is quoted against "
       f"{ceil16['accuracy_pct']}% here. And these bound a *lookup table* - "
       f"chroma predicted from one 64-bin number per pixel - not a network, which sees full "
       f"resolution luminance and a neighbourhood; route B's validation cross-entropy is "
       f"allowed to land under {ceil16['ce_floor_nats']} nats and when it does, that is the "
       f"result, not a contradiction. The only hard ceiling here is the {s_bin['psnr_mean']:.2f} "
       f"dB of the binned truth above, which no decoder restricted to bin centres can pass.")

    # 11 ------------------------------------------------------------------------------------- #
    # Y is copied, never predicted, even when the head is not zero.
    with torch.no_grad():                       # a nonzero head, chroma up to ~8 of 128
        model_a.unet.head.weight.normal_(0, 0.05)
        model_a.unet.head.bias.normal_(0, 0.1)
    out = fitting.decode(model_a, valid[:20], alpha=1.0)
    y_in = cs.ycc(valid[:20])[0]
    y_out = cs.ycc(out["rgb"])[0]
    clipped = (out["rgb"] <= 0.5) | (out["rgb"] >= 254.5)      # (N, H, W, 3)
    free = ~clipped.any(-1)                                    # a pixel is free if no channel cut
    drift = np.abs(y_out - y_in)
    mag = max(float(np.abs(out["cb"]).max()), float(np.abs(out["cr"]).max()))
    assert drift[free].max() < 0.01, drift[free].max()
    assert free.mean() > 0.8, f"{(1 - free.mean()) * 100:.0f}% clipped: too much to mean anything"
    assert (1 - free.mean()) > 0.01, "no clipping at all, so the exclusion is vacuous"
    assert mag > 1.0, f"the head is producing chroma of {mag}: not a test of anything"
    ok("Y is copied, not predicted", f"with a deliberately non-zero head (|chroma| up to "
                                     f"{mag:.1f} of 128, colourfulness "
                                     f"{cs.colourfulness(out['cb'], out['cr']):.3f} against the "
                                     f"photos' own 0.0402), the decoded "
                                     f"photo's luminance equals the input Y on the "
                                     f"{free.mean() * 100:.1f}% of pixels where no channel was "
                                     f"cut, to a maximum drift of {drift[free].max():.2e} of "
                                     f"255; the other {(1 - free.mean()) * 100:.1f}% go out of "
                                     f"gamut, where the clip legitimately moves Y - the check "
                                     f"is non-vacuous in both directions")

    # 12 ------------------------------------------------------------------------------------- #
    controls = M.control_photos(*cs.ycc(valid))
    per_gray = M.baseline_per_photo(valid)
    for name, pred in controls.items():
        s = M.score(pred, valid, baseline_per_photo=per_gray)
        p = M.paired(M.psnr_photos(pred, valid), per_gray)
        # The bar is not "never wins a photo" - a control keeps the luminance, so it can win
        # the odd near-grey photo, and here it does. The bar is that the deficit is real:
        # the paired interval must exclude zero, lose more photos than it wins, and the whole
        # set must sit below gray. A control that is merely usually worse proves nothing.
        assert s["psnr_mean"] < s_gray["psnr_mean"], (name, s)
        assert p["ci95"][1] < 0, (name, p)
        assert p["wins_b"] > p["wins_a"], (name, p)
        ok(f"control: {name}", f"{s['psnr_mean']:.2f} dB against gray's "
                               f"{s_gray['psnr_mean']:.2f} - a paired deficit of "
                               f"{-p['mean']:.2f} dB (95% CI {abs(p['ci95'][1]):.2f} to "
                               f"{abs(p['ci95'][0]):.2f}, {p['in_sds']:.1f} sems), losing "
                               f"{p['wins_b']}/{p['n']} photos and winning "
                               f"{p['wins_a']}/{p['n']}; its colourfulness is "
                               f"{s['colourfulness']:.4f}, the same chroma as the truth itself, "
                               f"just in the wrong place. So the metric is not rewarding "
                               f"colourfulness as such")

    # 13 ------------------------------------------------------------------------------------- #
    ds = d6.Photos(train[:8], "A", d6.Config())
    x1, y1 = ds[0]
    x2, y2 = ds[0]
    assert torch.equal(x1, x2) and torch.equal(y1, y2), "an epoch must see the same photo twice"
    xf, yf = d6.Photos(train[:8], "A", d6.Config(flip=True))[1]
    assert torch.equal(xf, torch.from_numpy(d6.input_of(train[1][:, ::-1])))
    assert torch.equal(yf, torch.from_numpy(d6.target_of(train[1][:, ::-1], "A")))
    assert abs(cs.colourfulness(*cs.ycc(train[1][:, ::-1])[1:])
               - cs.colourfulness(*cs.ycc(train[1])[1:])) < 1e-9, "a flip must not change colour"
    xb, yb = d6.Photos(train[:8], "B", d6.Config())[3]
    assert x1.shape == (1, 64, 64) and x1.dtype == torch.float32 and 0.0 <= float(x1.min()) \
        and float(x1.max()) <= 1.0
    assert y1.shape == (2, 64, 64) and y1.dtype == torch.float32 and float(y1.abs().max()) <= 1.01
    assert yb.shape == (64, 64) and yb.dtype == torch.int64 and int(yb.min()) >= 0 \
        and int(yb.max()) < cs.GRID ** 2
    # The batch helpers must be the stacked single helpers, and both must refuse the wrong
    # rank: `input_of` on a stack used to return a (1, N, 64, 64) array, which is an
    # N-channel image to the network - it fails at the first conv, but only after the
    # batching bug has already been trained on for a session.
    assert np.array_equal(d6.inputs_of(train[:5]),
                          np.stack([d6.input_of(p) for p in train[:5]]))
    for route in ("A", "B"):
        assert np.array_equal(d6.targets_of(train[:5], route),
                              np.stack([d6.target_of(p, route) for p in train[:5]])), route
    for name, fn in (("input_of", lambda: d6.input_of(train[:5])),
                     ("target_of", lambda: d6.target_of(train[:5], "A"))):
        try:
            fn()
            raise AssertionError(f"{name} accepted a batch of photos")
        except ValueError as exc:
            assert "use input" in str(exc) or "use target" in str(exc), exc
    assert np.array_equal(d6.inputs_of(valid[:4]), x.numpy()), "the loader's x is inputs_of"
    ok("the datasets and no-augmentation", f"x {tuple(x1.shape)} {x1.dtype} in "
                                           f"[{float(x1.min()):.3f}, {float(x1.max()):.3f}]; "
                                           f"A target {tuple(y1.shape)} within +-1 "
                                           f"(|max| {float(y1.abs().max()):.3f}); B target "
                                           f"{tuple(yb.shape)} int64 in [0, {cs.GRID ** 2}); "
                                           f"two reads of one photo are identical, and the flip "
                                           f"arm flips x and y together and leaves colourfulness "
                                           f"alone to 1e-9; the batch helpers equal the stacked "
                                           f"single-photo helpers exactly and both single helpers "
                                           f"refuse a stack")

    # 14 ------------------------------------------------------------------------------------- #
    # The prior pull must be an inference-time operation and nothing else. Constructed, not
    # sampled: this model's p IS the prior at every pixel, so on its own logits the pull
    # changes nothing and the check would pass while proving nothing.
    #
    # Note the direction, because it is the reason B+ exists: log pbar is negative, so
    # subtracting gamma * log pbar is a BONUS that grows as the cell gets rarer. argmax p
    # collapses onto the modal cell - 26.7% of all training pixels are one cell, which is
    # why plain B comes out the colour of wet cardboard - and the pull is what pushes pixels
    # back off it and into the colourful rare bins.
    lg, gamma = np.log(prior), 0.3
    rare = int(np.argmin(prior))                      # the rarest cell (smoothed, never empty)
    common = int(np.argmax(prior))                    # the modal one
    bonus = gamma * (lg[common] - lg[rare])           # what the pull hands `rare` over `common`
    margin = 0.5 * bonus                              # half of it, so the pull wins by half
    table = np.full(cs.GRID ** 2, lg[common] - 20.0)
    table[common] = lg[common]
    table[rare] = lg[common] - margin                 # pre-pull, the modal cell wins
    plain, pulled = table.argmax(), (table - gamma * lg).argmax()
    assert plain == common and pulled == rare, (plain, pulled, margin)
    # A FRESH prior-initialised model, not `model_b`: check 8 randomised that head, so its p
    # is no longer the prior. On a model that has learned nothing, p = pbar everywhere, so
    # argmax (1 - gamma) log pbar is still the modal cell and the pull must be inert.
    fresh = fitting.Colorizer("B", [16, 32, 64], prior_prob=prior)
    before = {k: v.clone() for k, v in fresh.state_dict().items()}
    g0 = fitting.decode(fresh, valid[:8], prior, gamma=0.0)["bin"]
    g3 = fitting.decode(fresh, valid[:8], prior, gamma=gamma)["bin"]
    after = fresh.state_dict()
    same = all(torch.equal(before[k], after[k]) for k in before)
    assert same, "decode() moved the weights"
    assert int((g0 != g3).sum()) == 0, "the pull moved a prior-initialised model"
    assert int((g0 == common).sum()) == g0.size, "a prior-init model should pick cell 120 flat"
    ok("gamma is inference-only", f"on a constructed log p where the modal cell {common} "
                                  f"(p = {prior[common]:.3f}) leads the rarest cell {rare} "
                                  f"(p = {prior[rare]:.2e}) by {margin:.2f} nats, argmax p "
                                  f"keeps {common} and argmax (log p - {gamma} log pbar) hands "
                                  f"the pixel to {rare}: the pull does change decisions, and it "
                                  f"changes them towards rare bins. Note the size of the lever "
                                  f"- at gamma = {gamma} the rarest cell is worth {bonus:.1f} "
                                  f"nats more than the modal one, so gamma is a strong knob and "
                                  f"calibrate.py will choose it on a curve, not by taste. On "
                                  f"this untrained model, whose p is the prior at every pixel, "
                                  f"the pull moves {int((g0 != g3).sum())} of {g0.size} pixel "
                                  f"bins (0 is right here) and leaves all {len(before)} "
                                  f"parameter tensors bit-identical")

    # 15 ------------------------------------------------------------------------------------- #
    warm = 3
    tl = torch.utils.data.DataLoader(d6.Photos(train[:1280], "A", d6.Config()), batch_size=32)
    tb = torch.utils.data.DataLoader(d6.Photos(train[:1280], "B", d6.Config()), batch_size=32)
    times = {}
    for name, loader, model in (("A", tl, model_a), ("B", tb, model_b)):
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        gen = torch.Generator().manual_seed(0)
        for k, (x, y) in enumerate(loader):
            x, y = x.to(fitting.part1.device), y.to(fitting.part1.device)
            opt.zero_grad()
            if name == "A":
                loss = fitting.loss_a(model(x), y)
            else:
                feats = model.features(x)
                idxp = fitting.sample_pixels(len(x), 4096, 1024, gen)
                loss = fitting.loss_b_at(model.head(feats, idxp), y.reshape(-1)[idxp.cpu()])
            loss.backward()
            opt.step()
            if k < warm:
                continue
            if k == warm + 4:
                break
        t1 = time.perf_counter()
        for k, (x, y) in enumerate(loader):
            x, y = x.to(fitting.part1.device), y.to(fitting.part1.device)
            opt.zero_grad()
            if name == "A":
                loss = fitting.loss_a(model(x), y)
            else:
                feats = model.features(x)
                idxp = fitting.sample_pixels(len(x), 4096, 1024, gen)
                loss = fitting.loss_b_at(model.head(feats, idxp), y.reshape(-1)[idxp.cpu()])
            loss.backward()
            opt.step()
            if k == 4:
                break
        times[name] = (time.perf_counter() - t1) / 5
    per_epoch = {k: v * np.ceil(9469 / 32) for k, v in times.items()}
    assert all(v > 0 for v in times.values()) and all(v < 10 for v in times.values())
    ok("timing", f"{times['A']:.2f} s and {times['B']:.2f} s per training batch of 32 for route "
                 f"A and B at widths [16,32,64] (3 warm-up batches discarded), i.e. about "
                 f"{per_epoch['A']:.0f} s and {per_epoch['B']:.0f} s per epoch of the 9469 "
                 f"training photos on 6 CPU threads")

    # 16 ------------------------------------------------------------------------------------- #
    # A call to a helper with the wrong arguments is invisible in a result: it sits in a script
    # that has not been run yet, behind however many minutes of training come first. Binding
    # every call site is a second's work and needs nothing executed.
    #
    # The same holds for a bare module alias that the file never imported - `figs.FIGURES = tmp`
    # in a file that imports figs only through the MODS dict below. No amount of argument binding
    # sees that; it is a NameError waiting in the one code path that touches it, and check 17
    # found exactly one when it ran for the first time. So every attribute access on a bare name
    # is checked to name something the file actually binds: an import, a definition, an
    # assignment, a parameter, a comprehension or loop target, an except or with name. The test
    # is deliberately file-wide rather than scope-precise, so it can only ever say "this name is
    # nowhere in the file" - which is always worth knowing - and never "wrong scope".
    files = sorted(p for p in d6.HERE.glob("*.py") if not p.name.startswith("_"))
    trees = {p: ast.parse(p.read_text(encoding="utf-8"), filename=p) for p in files}
    bad, calls, loose = [], 0, []
    for path, tree in trees.items():
        alias = {a.asname or a.name: MODS[a.name]
                 for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
                 if a.name in MODS}
        bound = {a.asname or a.name.split(".")[0]
                 for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                 for a in n.names}
        bound |= {n.id for n in ast.walk(tree)
                  if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        bound |= {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
        bound |= {n.name for n in ast.walk(tree)
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                    ast.ExceptHandler)) and n.name}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                name = node.value.id
                if name not in bound and name not in dir(builtins):
                    loose.append(f"{path.name}:{node.lineno} {name}.{node.attr}")
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            mod = alias.get(getattr(node.func.value, "id", ""))
            fn = getattr(mod, node.func.attr, None) if mod else None
            if not callable(fn) or any(isinstance(a, ast.Starred) for a in node.args) \
                    or any(k.arg is None for k in node.keywords):
                continue                    # a splat cannot be bound without running the call
            try:
                inspect.signature(fn).bind(*[None] * len(node.args),
                                           **{k.arg: None for k in node.keywords})
            except (TypeError, ValueError) as e:
                bad.append(f"{path.name}:{node.lineno} {mod.__name__}.{node.func.attr}: {e}")
            calls += 1
    assert not bad, "calls that cannot bind:\n" + "\n".join(bad)
    assert not loose, "attribute access on a name the file never binds:\n" + "\n".join(loose)
    ok("every call binds", f"{calls} calls to this folder's own helpers across {len(files)} "
                           f"scripts, each bound against the real signature of the function it "
                           f"names (arguments not evaluated, so nothing ran), plus every "
                           f"`name.attribute` in those scripts against the imports the file "
                           f"actually has. This is not "
                           f"theoretical: `figs.legend_on` was declared with a leading argument "
                           f"none of its three call sites passed, and `train_task6.py` only said "
                           f"so two minutes in, after it had trained route A and written half "
                           f"its output")

    # 17 ------------------------------------------------------------------------------------- #
    # The figure code is the last thing the pipeline executes - after the trained model, after
    # the calibration, and in `evaluate_task6.py` it sits between the test split and its results
    # file. So it gets executed here, on synthetic arrays too small to matter, into a temporary
    # directory that is deleted below: a broken plot should cost a second of the smoke test and
    # not a 2-hour run's only look at the test data.
    tmp = Path(tempfile.mkdtemp(prefix="task6_figcheck_"))
    real_figures, figs.FIGURES = figs.FIGURES, tmp
    try:
        rng = np.random.default_rng(0)
        syn = rng.integers(0, 256, (8, 64, 64, 3), dtype=np.uint8)
        y_s, cb_s, cr_s = cs.ycc(syn)
        shrunk = cs.rgb_of(y_s, cb_s * 0.4, cr_s * 0.4)
        bin_idx = cs.bin_index(cb_s, cr_s, cs.GRID)
        b_cb, b_cr = cs.bin_centre(bin_idx, cs.GRID)
        pulled = cs.bin_centre(cs.bin_index(cb_s * 1.6, cr_s * 1.6, cs.GRID), cs.GRID)
        dec = {"gray": {"rgb": M.baseline_photos(syn)},
               "A": {"rgb": shrunk},
               "B": {"rgb": cs.rgb_of(y_s, b_cb, b_cr)},
               "B+": {"rgb": cs.rgb_of(y_s, *pulled)},
               "binned truth": {"rgb": M.binned_truth(syn, cs.GRID)}}
        per = {k: M.psnr_photos(v["rgb"], syn) for k, v in dec.items()}
        base_s = M.baseline_per_photo(syn)
        idx, labels = E.rule_photos(syn, n_random=2)
        E.photo_figure(dec, syn, idx, labels, "smoke synthetic")
        E.chroma_figure(dec, syn, "smoke synthetic")
        E.quintile_figure(E.quintile_rows(per, base_s, syn), "smoke synthetic")
        made = sorted(p.name for p in tmp.glob("*.png"))
        sizes = {p.name: p.stat().st_size for p in tmp.glob("*.png")}
        assert len(made) == 3, made
        assert all(v > 5000 for v in sizes.values()), sizes
        assert len(idx) == len(labels) == 3 + 2, (idx, labels)   # 3 by rule + n_random, checked
        ok("the figures draw", f"{', '.join(made)} out of {tmp.name} (deleted), "
                               f"{sizes[made[0]] // 1024}-{max(sizes.values()) // 1024} KB, drawn "
                               f"from 8 synthetic photos through the same three functions "
                               f"`evaluate_task6.py` calls after it scores the test split - "
                               f"including the chroma figure's two awkward cases: gray, whose "
                               f"chroma is a delta function at the origin and is therefore drawn "
                               f"off the shared log scale rather than allowed to flatten every "
                               f"other panel, and the rule-ET5 photo picker, which on these "
                               f"random photos finds no exactly-grey one and takes its "
                               f"fallback ({labels[2]})")
    finally:
        figs.FIGURES = real_figures
        for p in tmp.glob("*"):
            p.unlink()
        tmp.rmdir()

    # The count is checked, not asserted elsewhere: a quoted "16 checks" in another script that
    # has fallen behind the code is the kind of mistake a reader has no way to notice.
    assert _printed == CHECKS, (f"the header and the other scripts quote {CHECKS} checks; "
                               f"{_printed} printed")
    print(f"\nsmoke test passed: {_printed} checks.")
    print(f"({time.perf_counter() - t0:.0f} s; nothing was trained to produce any of the above, "
          f"and the two models built here were thrown away)")


if __name__ == "__main__":
    main()
