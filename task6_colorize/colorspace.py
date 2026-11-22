"""The colour transform, written out, with its inverse and its bin grid.

The task's whole argument is about what `Y` is and what is left over, so the two
functions are here in six lines rather than imported - `skimage` is not installed in
this environment, and a conversion nobody can read is a conversion nobody can check.

Conventions, which every other file in this folder takes from here:

* `ycc` returns **Y in [0, 255]** and **Cb, Cr centred on 0**, i.e. in [-128, 128].
  Neutral is (0, 0), not 128: the network's output is in these units, so "no colour"
  is the number 0 and a zero-initialised head is exactly the gray baseline.
* `rgb_of` clips to [0, 255] on the way out, like every JPEG decoder does. It is the
  exact inverse of `ycc` for in-gamut pixels; `roundtrip_worst()` measures the residue.
* The Y used as the network's *input* is the Y of the *same* photo that defines the
  target, so a prediction that leaves Y alone can lose nothing. `rgb_of` is only ever
  called with the input Y - never with a Y the network produced (spec item 2).
* `bin_index` / `bin_centre` are the audit's `quantize`: a `grid x grid` grid on
  (Cb, Cr) covering [-128, 128), cell centres at `(i + 0.5) * step - 128`. Reused
  unchanged so that the ceilings quoted in the report (31.54 dB of binned truth, and
  30.1% top-1 for a colour lookup table, both at grid 16) are computed on the same grid
  the network is trained and scored on. The audit's 18.8% is that lookup table at its own
  32x32 grid; accuracy is meaningless without the grid it was counted at.

Run `python colorspace.py` for the self-checks: round-trip residue, the gamut of the
chroma that RGB can actually reach, and that the bin round-trip lands on centres.
"""
from __future__ import annotations

import numpy as np

# JPEG / ITU-R BT.601 coefficients. The notebook's own data cell uses the first three
# for the luminance it displays, so the input the task defines is this Y.
Y_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)
CB_W = np.array([-0.168736, -0.331264, 0.5], dtype=np.float32)
CR_W = np.array([0.5, -0.418688, -0.081312], dtype=np.float32)

CHROMA_SCALE = 128.0          # the network works in chroma / 128, so its target is ~ +-0.13
GRID = 16                     # route B's bin grid: 16 x 16 = 256 cells (the audit's K = 256)


def ycc(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(N, H, W, 3) uint8 or float RGB -> Y, Cb, Cr float32. Cb/Cr neutral at 0."""
    x = rgb.astype(np.float32)
    return (x @ Y_W,
            x[..., 0] * CB_W[0] + x[..., 1] * CB_W[1] + x[..., 2] * CB_W[2],
            x[..., 0] * CR_W[0] + x[..., 1] * CR_W[1] + x[..., 2] * CR_W[2])


def rgb_of(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """The exact inverse of `ycc`, clipped to the displayable range.

    `y` must be the input luminance, never a predicted one: 99% of the signal is Y and
    a drifting Y would put the drift into the score instead of the colour.
    """
    return np.clip(np.stack([y + 1.402 * cr,
                             y - 0.344136 * cb - 0.714136 * cr,
                             y + 1.772 * cb], -1), 0, 255)


def bin_index(cb: np.ndarray, cr: np.ndarray, grid: int = GRID) -> np.ndarray:
    """The (Cb, Cr) cell index in [0, grid*grid), row-major on (Cb, Cr)."""
    step = 2 * CHROMA_SCALE / grid
    i = np.floor((cb + CHROMA_SCALE) / step).astype(np.int64).clip(0, grid - 1)
    j = np.floor((cr + CHROMA_SCALE) / step).astype(np.int64).clip(0, grid - 1)
    return i * grid + j


def bin_centre(index: np.ndarray, grid: int = GRID) -> tuple[np.ndarray, np.ndarray]:
    """The (Cb, Cr) at the middle of each cell index - what a classifier predicts."""
    step = 2 * CHROMA_SCALE / grid
    i = index // grid
    j = index % grid
    return (i + 0.5) * step - CHROMA_SCALE, (j + 0.5) * step - CHROMA_SCALE


def chroma_l1(cb_a, cr_a, cb_b, cr_b) -> float:
    """The task's "mean absolute error on the chroma channels": mean over *pixels* of
    (|dCb| + |dCr|) / 255.

    Note the asymmetry with `colourfulness`, which the task also asks for: this one sums
    the two channels, that one averages them. So a gray prediction scores
    `chroma_l1 == 2 x colourfulness(truth)` - 0.0804 against 0.0402 on this data, which is
    the identity to check when a number looks a factor of two out. (The audit's tables use
    the same convention, so the two reports are comparable without a rescaling.)
    """
    return float((np.abs(cb_a - cb_b).mean() + np.abs(cr_a - cr_b).mean()) / 255.0)


def colourfulness(cb: np.ndarray, cr: np.ndarray) -> float:
    """mean over pixels *and* channels of |chroma| / 255.

    This is the definition the task's "0 for gray, about 0.04 for real photos" picks
    out: on the test photos it gives 0.041, and 0.000 for gray, and no other common
    definition (mean radius 0.062, pooled std 0.089, Hasler-Susstrunk 0.138) comes
    within a factor of 1.5 of 0.04. See task6_data_audit.md section 3.
    """
    return float((np.abs(cb).mean() + np.abs(cr).mean()) / 2 / 255.0)


def colourfulness_photos(cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """(N,): `colourfulness` one photo at a time, same definition.

    A batch's `colourfulness` is the mean of this vector whenever the photos are the same
    size, which they are here, so the two definitions cannot disagree about the headline
    number while disagreeing about which photo is the dullest one - which is what the figure
    rule and the colourfulness quintiles need.
    """
    if cb.ndim != 3:
        raise ValueError(f"colourfulness_photos takes (N, H, W); got {cb.shape}")
    return (np.abs(cb).mean((1, 2)) + np.abs(cr).mean((1, 2))) / 2.0 / 255.0


# --------------------------------------------------------------------------- self-check --
def roundtrip_worst(rgb: np.ndarray) -> float:
    """The worst |rgb_of(ycc(rgb)) - rgb| over the given photos, in units of 255."""
    y, cb, cr = ycc(rgb)
    return float(np.abs(rgb_of(y, cb, cr).astype(np.float64) - rgb.astype(np.float64)).max())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("worst round-trip residue, random RGB   :",
          f"{roundtrip_worst(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)[None]):.5f} of 255")

    # every RGB corner: the transform must stay invertible where it is defined, and the
    # clipping must only bite where the reconstruction really leaves [0, 255]
    corners = np.array([(r, g, b) for r in (0, 255) for g in (0, 255) for b in (0, 255)], np.float32)
    y, cb, cr = ycc(corners)
    back = rgb_of(y, cb, cr)
    print("corner pixels reproduced               :",
          f"{np.abs(back - corners).max():.5f} of 255")
    print("chroma of the 8 RGB corners   (Cb, Cr) :",
          ", ".join(f"({a:.0f},{b:.0f})" for a, b in zip(cb, cr)))
    print("gamut: |Cb| and |Cr| actually reachable:",
          f"{np.abs(cb).max():.1f}, {np.abs(cr).max():.1f} of the 128 the grid spans")

    for grid in (6, 16, 32):
        idx = bin_index(cb, cr, grid)
        cb2, cr2 = bin_centre(idx, grid)
        step = 2 * CHROMA_SCALE / grid
        err = np.maximum(np.abs(cb - cb2), np.abs(cr - cr2)).max()
        print(f"grid {grid:>2}: corner bins {sorted(idx.tolist())}, "
              f"worst |chroma - its centre| {err:.2f} (must be <= {step / 2 * 1.001:.2f})")

    # the audit's ceiling figure, reproducible from this file alone
    fake = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)[None].astype(np.float32) * 1.0
    y, cb, cr = ycc(fake)
    cb2, cr2 = bin_centre(bin_index(cb, cr), GRID)
    print(f"bin round-trip on a random photo: worst |dCb| {np.abs(cb - cb2).max():.1f}, "
          f"|dCr| {np.abs(cr - cr2).max():.1f}; centre of cell 0 = "
          f"{bin_centre(np.array([0]), GRID)[0][0]:.1f} (expect -120.0 at grid 16)")
