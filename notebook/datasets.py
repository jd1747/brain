# src/datasets 코드 통합본

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.configs.config import PROCESSED_DIR
from src.utils.utils import DatasetParams, RandomSeed

# 1. train/val/test 데이터셋 분할 및 metadata.csv 저장

SAVE_ROOT = PROCESSED_DIR / "patches"
METADATA_PATH = PROCESSED_DIR / "metadata.csv"
SPLIT_RATIO = DatasetParams.split_ratio
META_FIELDNAMES = ["patient_id", "image_dir", "mask_dir", "split"]


def build_metadata(save_root: Path, split_ratio=SPLIT_RATIO, seed: int = RandomSeed.seed):
    assert abs(sum(split_ratio) - 1.0) < 1e-6, "분할 비율 합은 1이어야 함"

    patient_ids = sorted(p.name for p in save_root.iterdir() if p.is_dir())

    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])
    # 나머지는 test set으로 배정

    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train : n_train + n_val]
    test_ids = patient_ids[n_train + n_val :]

    rows = []
    skipped = []
    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        for pid in ids:
            image_dir = save_root / pid / "image"
            mask_dir = save_root / pid / "mask"

            if not any(image_dir.glob("*.npy")):
                skipped.append(pid)
                continue

            rows.append({
                "patient_id": pid,
                "image_dir": str(image_dir),
                "mask_dir": str(mask_dir),
                "split": split_name,
            })

    return rows, skipped


def write_metadata(rows: list[dict], out_path: Path):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


# 2. 데이터셋 클래스 정의 및 증강변환 지시

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


# 3. 실행: split_data.py만 해당
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=float, default=SPLIT_RATIO[0])
    parser.add_argument("--val", type=float, default=SPLIT_RATIO[1])
    parser.add_argument("--test", type=float, default=SPLIT_RATIO[2])
    parser.add_argument("--seed", type=int, default=RandomSeed.seed)
    args = parser.parse_args()

    rows, skipped = build_metadata(SAVE_ROOT, split_ratio=(args.train, args.val, args.test), seed=args.seed)
    write_metadata(rows, METADATA_PATH)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["split"]] = counts.get(r["split"], 0) + 1

    print(f"환자 수 (split별): {counts}")
    if skipped:
        print(f".npy 없어서 제외된 환자 (skip): {len(skipped)}명")
    print(f"metadata 저장 완료: {METADATA_PATH}")
