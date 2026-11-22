"""The two heads, their losses, and a training loop that picks its epoch on validation loss.

`part1.UNet` is used exactly as the notebook wrote it - `UNet(2, c_in, c_out, widths)`, whose
own head is already a 1×1 conv - so both routes are the specification's shape with different
`c_out`: route A gets `c_out=2` (Cb, Cr in units of +-128), route B gets `c_out=256` (logits
over a 16x16 grid). Nothing in `part1.py` is edited; this file wraps it, and `features()`
below is the copied `UNet.forward` stopped one layer early so that route B's loss can be
computed on sampled pixels (see `pixels`, under WHICH PIXELS).

WHY THE EPOCH IS SELECTED ON LOSS AND NOT ON PSNR. On this data the per-photo PSNR of a
route-A predictor is highest at the point where it predicts no colour at all - the audit
measured that maximum at alpha = 0 before any network existed - so selecting the epoch by
validation PSNR would select epoch 1 and train nothing. The loss is the criterion that
actually rewards colour; the PSNR/colourfulness curve of every epoch is still logged, and
`figures/training_curves.png` shows the two criteria disagreeing, which is the task's trap
drawn on the training run itself.

WHICH PIXELS. Route B's logits are 256 channels at full resolution, so a batch of 32 photos
is 32 x 256 x 4096 floats before the softmax. The loss is therefore computed on `pixels`
of the 4096 per photo, drawn uniformly with a seeded generator, and the 1x1 head is applied
to those pixels only - which is arithmetically the same conv, just not evaluated where
nothing reads it. That is an expectation-equivalent estimator of the same objective, not a
different objective, and it is treated as a choice: `validate_choices.py` measures a
full-pixel arm against a sampled arm and the report quotes both. Validation loss is always
computed over all 4096 pixels, in chunks, so the model-selection number is exact.

INITIALISATION, which is a specification point rather than a detail:
  route A - the 1x1 head is zeroed, so the first forward pass predicts chroma 0 and scores
            *exactly* the gray baseline. Smoke check 6 asserts it to 1e-6 dB; if it does not
            hold, either Y is not being copied back or the scale is wrong.
  route B - head weights zeroed and the bias set to log of the training bin marginal, so the
            model starts at "I know nothing but the prior" and its epoch-1 accuracy is
            comparable with the prior's own. Logits are then constant in the input, which
            smoke check 7 asserts.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import colorspace as cs
import col6_data as d6
import part1

RESULTS = d6.HERE / "results"


def threads(n: int = 6) -> None:
    """Every second quoted in this folder was measured with this cap, as Tasks 1 and 3 do."""
    torch.set_num_threads(n)


def widths_for(levels: int, base: int = 32) -> list[int]:
    """[base, 2*base, ...] - the shape of the notebook's `widths` argument."""
    return [base * 2 ** k for k in range(levels)]


def prior(train_rgb: np.ndarray, grid: int) -> np.ndarray:
    """The training marginal over the bins: what a classifier that sees nothing predicts."""
    _, cb, cr = cs.ycc(train_rgb)
    counts = np.bincount(cs.bin_index(cb, cr, grid).ravel(), minlength=grid * grid)
    return (counts + 1.0) / (counts.sum() + grid)        # +1 smoothing: no log(0) at init


class Colorizer(nn.Module):
    """`part1.UNet` with the head reachable on its own."""

    def __init__(self, route: str, widths: list[int], c_in: int = 1,
                 prior_prob: np.ndarray | None = None):
        super().__init__()
        if route not in ("A", "B", "B+"):
            raise ValueError(route)
        self.route, self.widths, self.c_in = route, list(widths), c_in
        self.grid = cs.GRID if route != "A" else 0
        c_out = 2 if route == "A" else self.grid * self.grid
        self.unet = part1.UNet(2, c_in, c_out, self.widths)
        self.zero_head(prior_prob)

    def zero_head(self, prior_prob: np.ndarray | None) -> None:
        with torch.no_grad():
            self.unet.head.weight.zero_()
            if self.route == "A":
                self.unet.head.bias.zero_()                      # chroma 0 == the gray baseline
            else:
                p = torch.as_tensor(np.log(prior_prob), dtype=torch.float32) \
                    if prior_prob is not None else torch.zeros(self.grid * self.grid)
                self.unet.head.bias.copy_(p - p.mean())

    def features(self, x):
        """The copied UNet's forward(), stopped before its head (see the module docstring)."""
        feats = self.unet.encoder(x)
        x = feats.pop()
        for up, decode in zip(self.unet.up, self.unet.decode):
            skip = feats.pop()
            x = up(x, output_size=skip.shape[2:])
            x = decode(torch.cat([skip, x], dim=1))
        return x

    def head(self, feats, pixel_index: torch.Tensor | None = None):
        """The 1x1 head, over every pixel or only the ones `pixel_index` names.

        The notebook's head is `Conv2d(C, K, 1)`, whose weight is (K, C, 1, 1); a 1x1
        convolution is a linear map on the channel vector, so reshaping it to (K, C) and
        calling `F.linear` is the *same* function - not an approximation of it. Smoke check 5
        asserts the two paths agree to 1e-5 at a random init, which is what makes the
        sampled-pixel loss legitimate rather than merely convenient.
        """
        if pixel_index is None:
            return self.unet.head(feats)
        b, c, h, w = feats.shape
        weight = self.unet.head.weight
        if weight.dim() > 2:
            weight = weight.reshape(weight.shape[0], -1)
        flat = feats.permute(0, 2, 3, 1).reshape(b * h * w, c)
        return F.linear(flat[pixel_index], weight, self.unet.head.bias)

    def forward(self, x):
        return self.unet.head(self.features(x))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------------- losses --
def loss_a(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 on the two chroma planes in /128 units. MSE would return the conditional mean;
    L1 returns a slightly less dull answer for 0.3 dB (audit section 5, spec loss row)."""
    return F.l1_loss(pred, target)


def sample_pixels(batch: int, hw: int, n: int, generator: torch.Generator) -> torch.Tensor:
    """`n` pixel positions per photo, as flat indices into (batch*hw)."""
    off = torch.randint(0, hw, (batch, n), generator=generator)
    return (torch.arange(batch, device=off.device)[:, None] * hw + off).reshape(-1)


def loss_b_at(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on the bin labels, unweighted (the spec's loss row for route B: the
    prior pull belongs at inference, not in the objective)."""
    return F.cross_entropy(logits, labels)


# ------------------------------------------------------------------------------------ fit --
def fit(model: Colorizer, train_set, valid_set, prior_prob: np.ndarray | None = None,
        epochs: int = 12, lr: float = 3e-3, batch_size: int = 32, pixels: int | None = 1024,
        seed: int = 0, score_every: int = 1, verbose: bool = True) -> tuple:
    """Adam, best-validation-loss epoch kept. Returns (model, info).

    `pixels=None` scores the loss on all 4096 pixels per training photo. The validation loss
    always uses all of them, in chunks, whatever `pixels` is.
    """
    model = model.to(part1.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=part1.num_workers,
                                               generator=torch.Generator().manual_seed(seed))
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size,
                                               num_workers=part1.num_workers)
    info = {"route": model.route, "widths": model.widths, "epochs": epochs, "lr": lr,
            "batch_size": batch_size, "pixels": pixels, "seed": seed,
            "n_parameters": model.n_parameters(), "selected_on": "validation loss",
            "curve": {"train_loss": [], "valid_loss": [], "valid_psnr": [],
                      "valid_colourfulness": [], "epoch_seconds": []}}
    best, best_epoch, best_state = float("inf"), 0, None
    t_start = time.perf_counter()
    for epoch in range(epochs):
        t0 = time.perf_counter()
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(part1.device), y.to(part1.device)
            opt.zero_grad()
            if model.route == "A":
                loss = loss_a(model(x), y)
            else:
                feats = model.features(x)
                idx = sample_pixels(len(x), y.shape[-1] * y.shape[-2], pixels, gen) \
                    if pixels else None
                logits = model.head(feats, idx)
                if idx is None:      # the full-pixel arm: the head returns an image, not rows
                    logits = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
                    labels = y.reshape(-1)
                else:
                    labels = y.reshape(-1)[idx.cpu()]
                loss = loss_b_at(logits, labels)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        v_loss = validate_loss(model, valid_loader, pixels=None)
        info["curve"]["train_loss"].append(round(float(np.mean(losses)), 5))
        info["curve"]["valid_loss"].append(round(float(v_loss), 5))
        if score_every and (epoch + 1) % score_every == 0:
            psnr, cf = quick_valid(model, valid_loader)
            info["curve"]["valid_psnr"].append(round(psnr, 3))
            info["curve"]["valid_colourfulness"].append(round(cf, 4))
        else:
            info["curve"]["valid_psnr"].append(None)
            info["curve"]["valid_colourfulness"].append(None)
        if v_loss < best:
            best, best_epoch, best_state = v_loss, epoch + 1, copy.deepcopy(model.state_dict())
        info["curve"]["epoch_seconds"].append(round(time.perf_counter() - t0, 1))
        if verbose:
            extra = ""
            if info["curve"]["valid_psnr"][-1] is not None:
                extra = (f"  valid PSNR {info['curve']['valid_psnr'][-1]:.2f} dB, colourful "
                         f"{info['curve']['valid_colourfulness'][-1]:.3f}")
            print(f"    epoch {epoch + 1:>3}/{epochs}  train {info['curve']['train_loss'][-1]:.5f}"
                  f"  valid {info['curve']['valid_loss'][-1]:.5f}{extra}"
                  f"  [{info['curve']['epoch_seconds'][-1]:.0f} s]", flush=True)
    model.load_state_dict(best_state)
    info["best_valid_loss"], info["best_epoch"] = round(best, 5), best_epoch
    info["seconds"] = round(time.perf_counter() - t_start, 1)
    if verbose:
        print(f"    {epochs} epochs in {info['seconds']:.0f} s; best validation loss "
              f"{best:.5f} at epoch {best_epoch}")
    return model, info


@torch.no_grad()
def validate_loss(model: Colorizer, valid_loader, pixels: int | None = None) -> float:
    """Validation loss over every pixel of every validation photo, in memory-sized chunks."""
    model.eval()
    tot, n = 0.0, 0
    for x, y in valid_loader:
        x, y = x.to(part1.device), y.to(part1.device)
        out = model(x)
        if model.route == "A":
            v = float(loss_a(out, y))
        else:
            labels = y.reshape(-1)
            logits = out.permute(0, 2, 3, 1).reshape(-1, out.shape[1])
            step = 1 << 18                                   # pixels per CE chunk
            v = float(torch.stack([loss_b_at(logits[k:k + step], labels[k:k + step])
                                   for k in range(0, len(labels), step)]).mean())
        tot += v * len(x)
        n += len(x)
    return tot / max(n, 1)


@torch.no_grad()
def quick_valid(model: Colorizer, valid_loader) -> tuple[float, float]:
    """Validation PSNR and colourfulness, from the same decode the report uses, both routes.

    Logged every epoch so that the disagreement between "what the loss likes" and "what the
    task's first metric likes" is on the record for this run rather than asserted from the
    audit - and so that route B's own failure mode, a model that has learned the prior and
    nothing else, is visible as a flat colourfulness line instead of inferred from a loss.
    """
    import metrics as M
    truth_all = getattr(valid_loader.dataset, "rgb", None)
    if truth_all is None and model.route != "A":
        raise ValueError("route B cannot rebuild its truth from bin labels; the dataset needs .rgb")
    psnr_sum, cf_sum, n, seen = 0.0, 0.0, 0, 0
    for x, y in valid_loader:
        x = x.to(part1.device)
        out = model(x)
        if model.route == "A":
            cb = out[:, 0].cpu().numpy() * cs.CHROMA_SCALE
            cr = out[:, 1].cpu().numpy() * cs.CHROMA_SCALE
        else:                                     # argmax p: gamma 0, the plain B decode
            cb, cr = cs.bin_centre(out.argmax(1).cpu().numpy(), model.grid)
        if truth_all is not None:
            truth = truth_all[seen:seen + len(x)]
            y255 = cs.ycc(truth)[0]
        else:                                     # the targets rebuild the photo to 1e-4 of 255
            y255 = (x.cpu().numpy() * 255.0)[:, 0]
            truth = cs.rgb_of(y255, y[:, 0].cpu().numpy() * cs.CHROMA_SCALE,
                              y[:, 1].cpu().numpy() * cs.CHROMA_SCALE)
        pred = cs.rgb_of(y255, cb, cr)
        psnr_sum += float(M.psnr_photos(pred, truth).mean()) * len(x)
        cf_sum += float(cs.colourfulness(*cs.ycc(pred)[1:])) * len(x)
        n += len(x)
        seen += len(x)
    return psnr_sum / n, cf_sum / n


# ---------------------------------------------------------------------------------- decode --
@torch.no_grad()
def decode(model: Colorizer, rgb: np.ndarray, prior_prob: np.ndarray | None = None,
           alpha: float = 1.0, gamma: float = 0.0, batch: int = 16) -> dict:
    """Photos (N,64,64,3 uint8) -> predicted photos, and the chroma / bin behind them.

    Route A multiplies its chroma by `alpha` (the one calibrated scalar). Route B takes
    `argmax(logits - gamma * log pbar)`: `gamma = 0` is route B, `gamma > 0` is B+, and the
    pull is applied here at inference only - the same term inside the loss trains a
    different and worse classifier (the spec's last-layer row).
    """
    model.eval()
    ds = d6.Photos(rgb, model.route, d6.Config())       # the targets are ignored; only x is used
    loader = torch.utils.data.DataLoader(ds, batch_size=batch)
    y, _, _ = cs.ycc(rgb)
    bias = -gamma * np.log(prior_prob) if gamma else None
    if gamma and prior_prob is None:
        raise ValueError("route B+ needs the training prior")

    if model.route == "A":
        planes = [model(x.to(part1.device)).cpu().numpy() for x, _ in loader]
        pred = np.concatenate(planes)
        cb, cr = pred[:, 0] * cs.CHROMA_SCALE * alpha, pred[:, 1] * cs.CHROMA_SCALE * alpha
        return {"rgb": cs.rgb_of(y, cb, cr), "cb": cb, "cr": cr}

    # Route B: the 256-way logits are 4 MB per photo, so the argmax happens inside the loop
    # and only the winning cell is kept - keeping the logits of 500 test photos is 2 GB.
    index = []
    for x, _ in loader:
        logits = model(x.to(part1.device)).cpu().numpy()
        logits = logits.transpose(0, 2, 3, 1)                       # (B, H, W, K)
        m = logits.max(-1, keepdims=True)
        logp = logits - m - np.log(np.exp(logits - m).sum(-1, keepdims=True))
        if bias is not None:
            logp = logp + bias[None, None, None, :]
        index.append(logp.argmax(-1))
    index = np.concatenate(index)
    cb, cr = cs.bin_centre(index, model.grid)
    return {"rgb": cs.rgb_of(y, cb, cr), "cb": cb, "cr": cr, "bin": index}


# ------------------------------------------------------------------ saved models, no pickle --
def save_model(model: Colorizer, path, prior_prob: np.ndarray | None = None, **meta) -> None:
    torch.save({"state_dict": model.state_dict(), "route": model.route,
                "widths": model.widths, "c_in": model.c_in,
                "prior": torch.as_tensor(prior_prob if prior_prob is not None else []),
                **meta}, path)
    print(f"  wrote {path.name}  ({model.n_parameters()} parameters, "
          f"route {model.route}, widths {model.widths})")


def load_model(path) -> tuple[Colorizer, dict]:
    blob = torch.load(path, weights_only=True)
    model = Colorizer(blob["route"], blob["widths"], blob.get("c_in", 1))
    model.load_state_dict(blob["state_dict"])
    prior = blob["prior"].numpy()
    return model, {**blob, "prior": prior if prior.size else None}
