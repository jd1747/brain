# python -m src.datasets.split_data

import argparse
import csv
import random
from pathlib import Path

from src.configs.config import PROCESSED_DIR
from src.utils.utils import DatasetParams, RandomSeed

SAVE_ROOT = PROCESSED_DIR / "patches"
METADATA_PATH = PROCESSED_DIR / "metadata.csv"

SPLIT_RATIO = DatasetParams.split_ratio
RANDOM_SEED = RandomSeed.seed

META_FIELDNAMES = ["patient_id", "image_dir", "mask_dir", "split"]


def build_metadata(save_root: Path, split_ratio=SPLIT_RATIO, seed: int = RANDOM_SEED):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=float, default=SPLIT_RATIO[0])
    parser.add_argument("--val", type=float, default=SPLIT_RATIO[1])
    parser.add_argument("--test", type=float, default=SPLIT_RATIO[2])
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
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
