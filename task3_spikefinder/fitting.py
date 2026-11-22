"""The training-side pieces Task 3 needs and `part1.py` does not contain.

`part1.py` is copied verbatim and stays verbatim, so it is *used* rather than edited:
`part1.UNet` is the encoder/decoder, `part1.train` is the loop the smoke test exercises.
Three things are added here.

**The head.** The spec's last layer is "1 x 1 convolution to one channel, then
softplus". `part1.UNet` already ends in the 1 x 1 convolution; `RateNet` adds the
softplus, so a rate is non-negative without touching the copied class. At
initialization the head outputs about 0, so the network starts by predicting
softplus(0) = 0.69 everywhere - and because the target is divided by each neuron's own
mean rate, that mean is 1.0, which makes 0.69 a nearly unbiased starting point rather
than an arbitrary one. Softplus rather than ReLU because 83-98% of the bins are empty:
ReLU has *no* gradient exactly where almost all of the data sits.

**The loss.** `poisson_nll` is the Poisson negative log-likelihood of the counts,
written for a rate expressed in units of the neuron's own mean rate (see
`sf_data`): with λ = μ·ρ and r = y/μ, the NLL divided by μ is `ρ - r·log ρ` up to a
constant that does not depend on the weights. So this is the plain Poisson likelihood
of the real counts, just renormalized per neuron - which is what makes the score's
equal weighting of the 174 neurons match the loss. It keeps the `loss_fn(model(x), y)`
signature `part1.train` expects, so it can be handed to the copied loop unchanged.

**The epoch that gets kept.** `part1.train` returns the weights of the best validation
*loss*. The challenge scores a correlation, and a Poisson loss and a Pearson
correlation do not necessarily peak in the same epoch - the loss cares about the level
of the rate, the score cannot even see the level. `fit()` is the notebook's loop with
that one line changed: it keeps the best validation **correlation** over whole
validation recordings, and logs the loss beside it so the two curves can be compared.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import metrics
import part1
import sf_data

EPS = 1e-6


class RateNet(nn.Module):
    """The task's network: part1's UNet, then softplus. `widths` is the number of
    channels at each encoder level, so len(widths) is the number of resolutions."""

    def __init__(self, widths, c_in: int = 1, c_out: int = 1):
        super().__init__()
        self.net = part1.UNet(1, c_in, c_out, widths)
        self.act = nn.Softplus()

    def forward(self, x):
        return self.act(self.net(x))

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def poisson_nll(pred: torch.Tensor, rel: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """mean over the batch of (mean over a window of (ρ - r·log ρ)).

    ρ is the network's rate in units of that neuron's own mean rate, r the same for the
    target counts. Averaging *within* the window first is the point: it makes one
    window of a quiet neuron count as much as one window of a loud one, which is how
    the challenge's per-neuron average behaves.
    """
    rho = pred.clamp_min(0.0) + eps               # softplus output, guarded for the log
    return (rho - rel * torch.log(rho)).mean(dim=-1).mean()


def widths_for(levels: int, base: int = 16) -> list:
    """The notebook's own width rule: [16, 32, 64, ...] with `levels` levels."""
    return [base * 2 ** k for k in range(levels)]


@torch.no_grad()
def predict_trace(model: nn.Module, x: np.ndarray) -> np.ndarray:
    """The predicted rate over a whole recording, in one forward pass.

    The network is fully convolutional, so a recording of any length is a legal input -
    which is why the windows exist only for batching during training and no stitching
    of overlapping windows is needed here. Predictions near the two ends see less
    context than those in the middle (the field of view is ~1.8 s); that is a property
    of the recording's ends, and it applies to the baselines in the comparison table in
    exactly the same way.
    """
    model.eval()
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None, None].to(part1.device)
    return model(t)[0, 0].cpu().numpy()


def predict_split(model: nn.Module, data: dict, split: str) -> dict:
    return {n: predict_trace(model, data["x"][n]) for n in data["split"][split]}


def correlation(model: nn.Module, data: dict, split: str) -> dict:
    """The challenge's score for one split, over whole recordings. The bin count comes
    from the data config, so a run that represents the signal at 50 or 100 Hz is still
    scored on the challenge's 40 ms bins - the metric never changes, only the grid the
    network works on does."""
    bins = data["cfg"].metric_bins
    preds = predict_split(model, data, split)
    return metrics.score({n: metrics.correlate(p, data["y"][n], bins)
                          for n, p in preds.items()})


def fit(model: nn.Module, train_set, data: dict, loss_fn=poisson_nll, epochs: int = 60,
        lr: float = 3e-3, batch_size: int = 32, score_every: int = 1,
        verbose: bool = True) -> tuple:
    """The notebook's training loop, selecting on the validation correlation.

    Returns (model holding the best-correlation weights, a per-epoch log). The best
    model is *reloaded* at the end, so what this returns is what was measured - the
    same guarantee part1.train makes, for a different criterion.
    """
    model = model.to(part1.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                        num_workers=part1.num_workers)
    valid_loader = DataLoader(data["valid_set"], batch_size=batch_size,
                              num_workers=part1.num_workers)
    log, best = [], -np.inf
    best_state = copy.deepcopy(model.state_dict())
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x.to(part1.device)), y.to(part1.device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses))

        model.eval()
        with torch.no_grad():
            vl = [float(loss_fn(model(x.to(part1.device)), y.to(part1.device)).item())
                  for x, y in valid_loader]
        row = {"epoch": epoch + 1, "train_loss": train_loss,
               "valid_loss": float(np.mean(vl)), "seconds": round(time.perf_counter() - start, 1)}
        if (epoch + 1) % score_every == 0 or epoch + 1 == epochs:
            s = correlation(model, data, "valid")
            row["valid_corr"] = round(s["mean_over_neurons"], 4)
            row["valid_corr_datasets"] = round(s["mean_over_datasets"], 4)
            if s["mean_over_neurons"] > best:
                best, best_state, row["best"] = s["mean_over_neurons"], \
                    copy.deepcopy(model.state_dict()), True
        log.append(row)
        if verbose:
            print(f"   epoch {epoch + 1:>3}/{epochs}  loss {train_loss:.4f} -> "
                  f"{row.get('valid_loss', float('nan')):.4f}"
                  f"   valid correlation {row.get('valid_corr', float('nan')):.3f}"
                  f"{'   <-- best' if row.get('best') else ''}   "
                  f"[{row['seconds']:.0f} s]")
    model.load_state_dict(best_state)
    if verbose:
        print(f"   best validation correlation {best:.3f}; trained in "
              f"{time.perf_counter() - start:.0f} s")
    return model, {"best_valid_corr": best, "epochs": epochs, "log": log,
                   "lr": lr, "batch_size": batch_size}


def load_model(widths, path, c_in: int = 1) -> nn.Module:
    """Rebuild the architecture and put the saved weights in it. Only weights are in
    `model.pt` (no pickle of a class), so the architecture is part of the run's
    configuration, recorded in results/choices.json."""
    model = RateNet(widths, c_in)
    model.load_state_dict(torch.load(path, map_location=part1.device)["state_dict"])
    return model.eval()
