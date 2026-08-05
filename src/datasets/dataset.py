from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

AUG_COMBINATIONS = [
    {"flip": None, "rotation": 0},
    {"flip": "horizontal", "rotation": 0},
    {"flip": None, "rotation": 90},
    {"flip": None, "rotation": -90},
    {"flip": "horizontal", "rotation": 90},
]

_NO_AUG = [{"flip": None, "rotation": 0}]


def _apply_augmentation(image: np.ndarray, mask: np.ndarray, combo: dict):
    if combo["flip"] == "horizontal":
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    elif combo["flip"] == "vertical":
        image = np.flipud(image)
        mask = np.flipud(mask)

    if combo["rotation"] != 0:
        image = np.rot90(image, k=1)
        mask = np.rot90(mask, k=1)

    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class BratsSliceDataset(Dataset):
    def __init__(self, metadata_csv: Path, split: str, augment: bool | None = None):
        assert split in {"train", "val", "test"}
        self.split = split

        if augment is None:
            augment = split == "train"
        if split != "train" and augment:
            raise ValueError("val/test data에는 증강 미적용")
        self.augment = augment

        df = pd.read_csv(metadata_csv)
        df = df[df["split"] == split]

        combos = AUG_COMBINATIONS if self.augment else _NO_AUG

        self.samples: list[tuple[Path, Path, dict]] = []
        for _, row in df.iterrows():
            image_dir = Path(row["image_dir"])
            mask_dir = Path(row["mask_dir"])

            for image_path in sorted(image_dir.glob("*.npy")):
                mask_path = mask_dir / image_path.name
                if not mask_path.exists():
                    continue
                for combo in combos:
                    self.samples.append((image_path, mask_path, combo))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, mask_path, combo = self.samples[idx]

        image = np.load(image_path)
        mask = np.load(mask_path)

        if combo["flip"] is not None or combo["rotation"] != 0:
            image, mask = _apply_augmentation(image, mask, combo)

        image = image[np.newaxis, ...].astype(np.float32)
        mask = mask[np.newaxis, ...].astype(np.float32)

        return image, mask
