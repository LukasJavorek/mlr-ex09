"""Task 6's loader: Imagenette2-160 -> the notebook's 64x64 photos -> torch datasets.

`load_photo` and the split are **copied verbatim from the notebook's Task 6 data cell**
(cell 76, id `b221b280`), including `sorted(glob)`, `Image.BICUBIC` with no pre-filter,
the central 64x64 and the `default_rng(0)` permutation of `val/`. That is deliberate: the
audit measured the data the *cell* produces - the aliasing floor, the 75% median crop, the
split sizes - and a loader that quietly "improved" the resampling would invalidate every
ceiling the report quotes. `../task6_data_audit.md` section 8 argues the case for keeping
it; the short version is that the same error sits in the input and in the target, so it
costs no PSNR, and changing it moves the baseline too.

Three differences from the cell, none of which touches a pixel:

* the 13,394 decodes are cached once in `data/cache_s64.npz` (a cold build is ~1 min, every
  later run seconds). Only the decoded pixels and the file names are cached - the split is
  recomputed from `cfg.seed` on every load, so a wrong cache cannot quietly survive a
  change of seed;
* the **3325 photos of `val/` that the cell hands to nobody** are loaded too, as the split
  `valrest`. They are a confirmation pool only: no choice in this folder is made on them,
  and they are never mixed into the same number as the handout's 100 (spec item 7);
* the file names are kept, so any reported photo can be traced back to its JPEG.

    python col6_data.py            # build the cache and print the inventory

The inventory ends with a byte-for-byte cross-check against the audit's arrays, because
every ceiling in the report was measured on those: "the model trained on the pixels the
audit measured" is checked, not asserted.
"""
from __future__ import annotations

import json
import sys
import tarfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import colorspace as cs

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
COL = DATA_DIR / "imagenette2-160"
COL_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
SIZE = 64
N_TRAIN_EXPECT, N_VAL_EXPECT = 9469, 3925

# The audit read the same photos through the same code; if that cache is on disk this
# folder cross-checks itself against it (`audit_crosscheck`, and smoke check 2).
AUDIT_CACHE = HERE.parent / "_analysis" / "col_cache"


@dataclass
class Config:
    """Everything the arrays depend on. `size` is the notebook's; the rest are choices.

    `flip` is the one augmentation this folder is willing to consider: a horizontal flip
    moves colour together with the geometry, which is what colour is supposed to do. It is
    measured as an arm (`validate_choices.stage_two`) rather than assumed. Colour jitter
    and friends are not on the table at all - they edit the target (spec item 8).
    """
    size: int = SIZE
    seed: int = 0
    grid: int = cs.GRID
    flip: bool = False

    def key(self) -> str:
        return f"s{self.size}"


def seed_everything(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_photo(path, size=SIZE):
    """The short side resized to 64 pixels, then the central 64 x 64 square.

    VERBATIM from the notebook, BICUBIC and all. Do not add a pre-filter, a random crop or
    a resize of your own: see this file's header and audit section 8.
    """
    img = Image.open(path).convert("RGB")
    scale = size / min(img.size)
    img = img.resize((max(size, round(img.width * scale)), max(size, round(img.height * scale))), Image.BICUBIC)
    left, top = (img.width - size) // 2, (img.height - size) // 2
    return np.asarray(img.crop((left, top, left + size, top + size)))


def ensure_imagenette(root: Path = DATA_DIR) -> Path:
    """The same download the notebook's cell does, into this folder's own `data/`."""
    col = root / "imagenette2-160"
    if len(list(col.glob("*/*/*.JPEG"))) != N_TRAIN_EXPECT + N_VAL_EXPECT:
        root.mkdir(exist_ok=True)
        print(f"downloading imagenette2-160 (95 MB) into {root}")
        urllib.request.urlretrieve(COL_URL, root / "imagenette2-160.tgz")
        with tarfile.open(root / "imagenette2-160.tgz") as archive:
            archive.extractall(root)
        (root / "imagenette2-160.tgz").unlink()
    return col


def split_of(names: dict[str, list[str]], seed: int = 0) -> dict[str, list[str]]:
    """The notebook's split, verbatim, plus `valrest`: what `order[600:]` leaves over.

    Recomputed on every load from `seed` - never cached - so that changing the seed cannot
    leave a stale split behind. The two parts arrive as separate lists rather than being
    re-parsed out of the paths: `str(Path)` yields a backslash on Windows and a slash on
    Linux, and a split that depends on a path separator is a split that changes with the
    operating system. (It cost this loader one run to learn that.)
    """
    train_files = sorted(names["train"])
    val_files = sorted(names["val"])
    order = np.random.default_rng(seed).permutation(len(val_files))
    return {"train": train_files,                       # 9469, everything the fit sees
            "valid": [val_files[k] for k in order[500:600]],    # the handout's 100: decides
            "test": [val_files[k] for k in order[:500]],        # read once, by evaluate_task6
            "valrest": [val_files[k] for k in order[600:]]}     # the 3325 nobody uses


def locate() -> Path:
    """Where the photos are: this folder's `data/`, else the repo's download dir, else fetch.

    The repo already holds one untarred copy at `../_imagenette_dl/imagenette2-160` (the
    audit's), and the pixels are the same 13,394 files either way - `audit_crosscheck`
    proves the arrays agree, so which copy was read is a matter of disk, not of result.
    """
    for cand in (COL, HERE.parent / "_imagenette_dl" / "imagenette2-160"):
        if len(list(cand.glob("*/*/*.JPEG"))) == N_TRAIN_EXPECT + N_VAL_EXPECT:
            return cand
    return ensure_imagenette(DATA_DIR)


def build(cfg: Config = Config(), base: Path | None = None, verbose: bool = True) -> dict:
    """Decode every photo once into `data/cache_s{size}.npz` and return the loaded data."""
    DATA_DIR.mkdir(exist_ok=True)
    base = base or locate()
    cache = DATA_DIR / f"cache_{cfg.key()}.npz"
    if not cache.exists():
        print(f"decoding {base}")
        files = sorted(str(p.relative_to(base)).replace("\\", "/") for p in base.glob("*/*/*.JPEG"))
        t0 = time.perf_counter()
        arrays = {}
        for part, n in (("train", N_TRAIN_EXPECT), ("val", N_VAL_EXPECT)):
            names = [f for f in files if f.startswith(f"{part}/")]
            assert len(names) == n, f"{part}: {len(names)} JPEGs, expected {n}"
            arrays[f"{part}_rgb"] = np.stack([load_photo(base / f, cfg.size) for f in names])
            arrays[f"{part}_files"] = np.array(names)
            print(f"  decoded {part:>5} {arrays[f'{part}_rgb'].shape} in "
                  f"{time.perf_counter() - t0:.0f} s")
        np.savez(cache, **arrays)
    with np.load(cache) as loaded:                      # the cache is only pixels + names
        rgb = {p: np.asarray(loaded[f"{p}_rgb"]) for p in ("train", "val")}
        names = {p: [str(x) for x in loaded[f"{p}_files"]] for p in ("train", "val")}
    out = _assemble(cfg, rgb, names, str(cache))
    if verbose:
        print("data: " + out["source"])
        print(summary(out))
    return out


def _assemble(cfg: Config, rgb: dict[str, np.ndarray], names: dict[str, list[str]],
              source: str) -> dict:
    split = split_of(names, cfg.seed)
    lookup = {n: (part, i) for part in ("train", "val")
              for i, n in enumerate(names[part])}       # name -> (part, row in that array)
    index = {s: [lookup[n] for n in files] for s, files in split.items()}
    return {"source": source, "cfg": asdict(cfg), "split": split, "index": index,
            "rgb": rgb, "files": names}


def arrays(data: dict, split: str) -> np.ndarray:
    """The (N, size, size, 3) uint8 photos of one split.

    `train` is returned as a view of the loaded cache (its rows are the decode order, so no
    116 MB copy per call); the val slices come out in the permutation's order and are copied.
    """
    rows = data["index"][split]
    shape = data["rgb"]["train"].shape
    if rows and rows[0][0] == "train" and [i for _, i in rows] == list(range(len(rows))):
        return data["rgb"]["train"]
    if rows and all(p == rows[0][0] for p, _ in rows):
        return data["rgb"][rows[0][0]][[i for _, i in rows]]
    out = np.empty((len(rows), shape[1], shape[2], 3), np.uint8)
    for k, (p, i) in enumerate(rows):
        out[k] = data["rgb"][p][i]
    return out


def exactly_grey(rgb: np.ndarray) -> np.ndarray:
    """Which photos have R == G == B in every pixel - what the baseline scores 120 dB on.

    3 of the 100 valid and 8 of the 500 test photos are source photographs that were black
    and white to begin with. The gray baseline reproduces them to within a float rounding,
    so their PSNR is a property of the epsilon in the code, not of the picture; that is why
    the metric clips each photo at a stated dB, and why the figure grid is asked to show one
    of them (spec item 5: "leaves black-and-white photos alone" is a behaviour worth seeing).
    """
    r = rgb.astype(np.int16)
    return ((r[..., 0] == r[..., 1]) & (r[..., 1] == r[..., 2])).reshape(len(rgb), -1).all(1)


def files_of(data: dict, split: str) -> list[str]:
    return data["split"][split]


def summary(data: dict) -> str:
    """What the loader prints, so the same numbers are in every log."""
    lines = []
    for split in ("train", "valid", "test", "valrest"):
        rgb = arrays(data, split)
        _, cb, cr = cs.ycc(rgb)
        lines.append(f"  {split:>7} {str(rgb.shape):>18}  sha1 {sha(rgb)}  "
                     f"colourfulness {cs.colourfulness(cb, cr):.4f}")
    return "\n".join(lines)


def sha(rgb: np.ndarray) -> str:
    import hashlib
    return hashlib.sha1(np.ascontiguousarray(rgb).tobytes()).hexdigest()[:12]


def configured(cfg: Config | None = None) -> Config:
    """The `Config` the sweep chose, if it ran; the notebook's defaults otherwise."""
    path = HERE / "results" / "choices.json"
    cfg = cfg or Config()
    if path.exists():
        return Config(**json.loads(path.read_text(encoding="utf-8"))["chosen"]["data"])
    return cfg


def audit_crosscheck(data: dict, splits=("train", "valid", "test")) -> list[str]:
    """This folder's arrays against the audit's, byte for byte. Raises on any disagreement."""
    lines = []
    for split in splits:
        theirs = AUDIT_CACHE / f"{split}.npy"
        if not theirs.exists():
            lines.append(f"  {split:>7}: the audit cache is not on disk, not compared")
            continue
        mine, theirs = arrays(data, split), np.load(theirs)
        if mine.shape != theirs.shape or not np.array_equal(mine, theirs):
            n = -1 if mine.shape != theirs.shape else int(
                (mine.reshape(len(mine), -1) != theirs.reshape(len(theirs), -1)).any(1).sum())
            raise AssertionError(f"{split}: differs from the audit cache "
                                 f"({mine.shape} vs {theirs.shape}, {n} photos disagree)")
        lines.append(f"  {split:>7}: {mine.shape} identical to the audit's, sha1 {sha(mine)}")
    return lines


# ------------------------------------------------------------------------------- targets --
def input_of(rgb: np.ndarray) -> np.ndarray:
    """(1, 64, 64) float32 in [0, 1]: the luminance, the only thing the net may look at.

    One photo. Passing a stack of them here would quietly produce a (1, N, 64, 64) array -
    an N-channel image to the network, which fails at the first conv rather than silently -
    so the shape is asserted and `inputs_of` below is for batches.
    """
    if rgb.ndim != 3:
        raise ValueError(f"input_of takes ONE photo, (64, 64, 3); got {rgb.shape} - "
                         f"use inputs_of for a batch")
    return (cs.ycc(rgb)[0] / 255.0)[None].astype(np.float32)


def inputs_of(rgb: np.ndarray) -> np.ndarray:
    """(N, 1, 64, 64) float32: `input_of` over a batch."""
    if rgb.ndim != 4:
        raise ValueError(f"inputs_of takes a batch, (N, 64, 64, 3); got {rgb.shape}")
    return (cs.ycc(rgb)[0] / 255.0)[:, None].astype(np.float32)


def target_of(rgb: np.ndarray, route: str, grid: int = cs.GRID):
    """What the network is asked for, from the photo.

    route "A"      -> (2, 64, 64) float32, (Cb, Cr) / 128, neutral at 0
    route "B"/"B+" -> (64, 64) int64, the `grid` x `grid` (Cb, Cr) bin index - at the default
                      grid 16, the same grid whose binned-truth ceiling (31.54 dB) and lookup
                      accuracy ceiling (30.1%) the report quotes. The audit's 18.8% belongs to
                      its 32x32 grid, not to this one
    """
    if rgb.ndim != 3:
        raise ValueError(f"target_of takes ONE photo, (64, 64, 3); got {rgb.shape} - "
                         f"use targets_of for a batch")
    _, cb, cr = cs.ycc(rgb)
    if route == "A":
        return (np.stack([cb, cr], 0) / cs.CHROMA_SCALE).astype(np.float32)
    if route in ("B", "B+"):
        return cs.bin_index(cb, cr, grid)
    raise ValueError(route)


def targets_of(rgb: np.ndarray, route: str, grid: int = cs.GRID):
    """(N, 2, 64, 64) float32 or (N, 64, 64) int64: `target_of` over a batch."""
    if rgb.ndim != 4:
        raise ValueError(f"targets_of takes a batch, (N, 64, 64, 3); got {rgb.shape}")
    _, cb, cr = cs.ycc(rgb)
    if route == "A":
        return (np.stack([cb, cr], 1) / cs.CHROMA_SCALE).astype(np.float32)
    if route in ("B", "B+"):
        return cs.bin_index(cb, cr, grid)
    raise ValueError(route)
    raise ValueError(route)


class Photos(Dataset):
    """Yields (luminance, target). No colour augmentation, on purpose.

    Colour jitter, channel shuffle, autocontrast and solarise all edit the thing being
    predicted; on this task they would be editing the answer, and the audit's ceilings
    would stop bounding the model. `cfg.flip` (a horizontal flip of input and target
    together) is the one augmentation allowed, and only because it is measured as an arm
    rather than assumed.
    """

    def __init__(self, rgb: np.ndarray, route: str, cfg: Config | None = None):
        self.rgb, self.route, self.cfg = rgb, route, cfg or Config()
        if self.rgb.shape[1:3] != (self.cfg.size, self.cfg.size):
            raise ValueError(f"expected {self.cfg.size}x{self.cfg.size}, got {self.rgb.shape}")
        if route not in ("A", "B", "B+"):
            raise ValueError(route)
        self.flip = bool(self.cfg.flip)

    def __len__(self) -> int:
        return len(self.rgb)

    def __getitem__(self, i: int):
        photo = self.rgb[i]
        if self.flip and (i % 2):                     # half the photos, deterministically:
            photo = photo[:, ::-1]                    # a flip of every photo is no augmentation
        x = torch.from_numpy(input_of(photo))
        t = target_of(photo, self.route, self.cfg.grid)
        return x, torch.from_numpy(t if self.route == "A" else t.astype(np.int64))


def sets(data: dict, route: str, splits=("train", "valid", "test")) -> dict[str, Photos]:
    """`Photos` datasets for the named splits only.

    A sweep calls this with ("train", "valid"): nothing in it can score, average or
    threshold a test photo if no test photo was materialised. `evaluate_task6.py` is the
    only caller that ever passes "test".
    """
    return {s: Photos(arrays(data, s), route, Config(**data["cfg"])) for s in splits}


# --------------------------------------------------------------------------------- main --
def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    t0 = time.perf_counter()
    data = build()
    print()
    for split in ("train", "valid", "test", "valrest"):
        rgb = arrays(data, split)
        y, cb, cr = cs.ycc(rgb)
        gray = cs.rgb_of(y, np.zeros_like(cb), np.zeros_like(cr))
        diff = np.abs(gray.astype(np.float64) - rgb.astype(np.float64))
        print(f"  {split:>7} Y mean {y.mean():5.1f}   colourfulness "
              f"{cs.colourfulness(cb, cr):.4f}   gray-vs-photo {diff.mean() / 255:.4f}/1 "
              f"({diff.mean():5.2f} of 255)   {int(exactly_grey(rgb).sum()):>3} photos black and "
              f"white already   e.g. {files_of(data, split)[0]}")
    print("\ncross-check against the audit's arrays:")
    print("\n".join(audit_crosscheck(data)))
    print("\n'gray-vs-photo' is mean |rgb_of(Y, 0, 0) - photo| in units of 1: about 0.05 on "
          "every split, ~13 of 255. It is 0 only where the photo was already black and white. "
          "Y is copied back unchanged, so a colouriser competes for the chroma and nothing else.")
    print("the black-and-white photos are why the PSNR has to be clipped at a stated dB "
          "(audit section 2): the baseline is perfect on them, and 1/log(0) is the code's epsilon.")
    print(f"\nsplits {({s: len(v) for s, v in data['split'].items()})}, seed {data['cfg']['seed']}")
    print(f"({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
