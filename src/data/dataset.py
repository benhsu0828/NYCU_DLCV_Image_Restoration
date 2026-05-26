import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_NAME_RE = re.compile(r"^(rain|snow)-(\d+)\.png$", re.IGNORECASE)


def _scan_pairs(data_root: Path):
    degraded_dir = data_root / "train" / "degraded"
    clean_dir = data_root / "train" / "clean"
    pairs = {"rain": [], "snow": []}
    for p in sorted(degraded_dir.iterdir()):
        m = _NAME_RE.match(p.name)
        if not m:
            continue
        deg_type, idx = m.group(1).lower(), int(m.group(2))
        clean_path = clean_dir / f"{deg_type}_clean-{idx}.png"
        if not clean_path.exists():
            raise FileNotFoundError(f"missing clean image for {p.name}: {clean_path}")
        pairs[deg_type].append((p, clean_path, idx))
    for k in pairs:
        pairs[k].sort(key=lambda x: x[2])
    return pairs


def build_train_val_splits(data_root: str | Path, val_per_type: int = 100):
    data_root = Path(data_root)
    pairs = _scan_pairs(data_root)
    train, val = [], []
    for deg_type, items in pairs.items():
        if len(items) <= val_per_type:
            raise ValueError(f"not enough samples for {deg_type}: {len(items)}")
        train.extend([(d, c, deg_type) for d, c, _ in items[:-val_per_type]])
        val.extend([(d, c, deg_type) for d, c, _ in items[-val_per_type:]])
    return train, val


def _load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


class PairedRestoreDataset(Dataset):
    """Returns (degraded, clean) tensors in [0,1]."""

    def __init__(self, items, patch_size: int | None = 128, train: bool = True):
        self.items = items
        self.patch_size = patch_size
        self.train = train

    def __len__(self):
        return len(self.items)

    def _augment(self, deg: np.ndarray, clean: np.ndarray):
        ps = self.patch_size
        h, w, _ = deg.shape
        if ps is not None:
            top = random.randint(0, h - ps)
            left = random.randint(0, w - ps)
            deg = deg[top : top + ps, left : left + ps]
            clean = clean[top : top + ps, left : left + ps]
        if random.random() < 0.5:
            deg = deg[:, ::-1].copy()
            clean = clean[:, ::-1].copy()
        if random.random() < 0.5:
            deg = deg[::-1, :].copy()
            clean = clean[::-1, :].copy()
        k = random.randint(0, 3)
        if k:
            deg = np.rot90(deg, k).copy()
            clean = np.rot90(clean, k).copy()
        return deg, clean

    def __getitem__(self, idx):
        deg_path, clean_path, _ = self.items[idx]
        deg = _load_rgb(deg_path)
        clean = _load_rgb(clean_path)
        if self.train:
            deg, clean = self._augment(deg, clean)
        deg_t = torch.from_numpy(deg).permute(2, 0, 1).float() / 255.0
        clean_t = torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0
        return deg_t, clean_t
