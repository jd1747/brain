import glob
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import (
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandBiasFieldd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
from monai.transforms.spatial.dictionary import Rand2DElasticd, RandFlipd, RandRotated, RandZoomd
from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped
from torch.utils.data import Dataset

from src.configs.config import METADATA_CSV_PATH


class BratsSliceDataset(Dataset):
    def __init__(
        self, metadata_csv: Path, split: str, transform: Callable | None = None, drop_empty_slices: bool = False
    ):
        self.transform = transform
        self.drop_empty_slices = drop_empty_slices

        df = pd.read_csv(metadata_csv)
        df = df[df["split"] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"split='{split}'에 해당하는 행이 metadata.csv에 없음")

        self.samples: list[tuple[str, str, str]] = []
        for _, row in df.iterrows():
            self.samples.extend(self._match_pairs(row["image_dir"], row["mask_dir"], row["patient_id"]))

        if len(self.samples) == 0:
            raise RuntimeError(f"split='{split}'에서 슬라이스를 찾을 수 없음")

        self._has_tumor_cache: list[bool] | None = None

    @staticmethod
    def _match_pairs(image_dir: str, mask_dir: str, patient_id: str) -> list[tuple[str, str, str]]:
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.npy")))
        mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.npy")))

        if len(image_files) != len(mask_files):
            raise RuntimeError(
                f"[{patient_id}]: image 슬라이스 수({len(image_files)})와 mask 슬라이스 수({len(mask_files)})가 일치하지 않음"
            )

        return [(img, mask, patient_id) for img, mask in zip(image_files, mask_files)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, mask_path, patient_id = self.samples[idx]

        image = np.load(image_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.uint64)

        if self.drop_empty_slices and mask.max() == 0:
            pass

        data = {"image": image, "label": mask}

        if self.transform is not None:
            data = self.transform(data)
        else:
            data["image"] = torch.from_numpy(image[None, ...])  # (1, H, W)
            data["label"] = torch.from_numpy(mask[None, ...])

        data["patient_id"] = patient_id
        return data

    def filter_empty_slices(self):
        kept = []
        for image_path, mask_path, patient_id in self.samples:
            mask = np.load(mask_path)
            if mask.max() > 0:
                kept.append((image_path, mask_path, patient_id))
        removed = len(self.samples) - len(kept)
        self.samples = kept

    def get_sample_weights(self, empty_weight: float = 0.2, tumor_weight: float = 1.0) -> torch.Tensor:
        weights = []
        for _, mask_path, _ in self.samples:
            mask = np.load(mask_path)
            weights.append(tumor_weight if mask.max() > 0 else empty_weight)
        return torch.tensor(weights, dtype=torch.double)


def build_train_transforms(
    rotate_range_deg: float = 30.0, scale_range: float = 0.2, elastic_prob: float = 0.2, flip_prob: float = 0.5
):
    rotate_rad = np.deg2rad(rotate_range_deg)

    return Compose([
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        # --- spatial ---
        RandFlipd(keys=["image", "label"], prob=flip_prob, spatial_axis=1),  # 좌우(width축)만
        RandRotated(
            keys=["image", "label"],
            range_x=rotate_rad,
            prob=0.5,
            mode=["bilinear", "nearest"],
            padding_mode="zeros",
        ),
        RandZoomd(
            keys=["image", "label"],
            min_zoom=1 - scale_range,
            max_zoom=1 + scale_range,
            prob=0.5,
            mode=["bilinear", "nearest"],
        ),
        Rand2DElasticd(
            keys=["image", "label"],
            spacing=(20, 20),
            magnitude_range=(1, 3),  # 과도한 변형 방지 위해 낮게 설정
            prob=elastic_prob,
            mode=["bilinear", "nearest"],
            padding_mode="zeros",
        ),
        # --- intensity (mask에는 적용 안 함) ---
        RandGaussianNoised(keys=["image"], prob=0.2, std=0.02),
        RandGaussianSmoothd(keys=["image"], prob=0.15),
        RandScaleIntensityd(keys=["image"], factors=0.2, prob=0.3),  # contrast-like
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),  # brightness-like
        RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.5), prob=0.3),
        RandBiasFieldd(keys=["image"], prob=0.2, coeff_range=(0.0, 0.3)),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ])


def build_val_transforms() -> Compose:
    """검증/테스트용: augmentation 없이 정규화만."""
    return Compose([
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ])


if __name__ == "__main__":
    # 간단한 동작 확인용

    train_ds = BratsSliceDataset(
        metadata_csv=METADATA_CSV_PATH,
        split="train",
        transform=build_train_transforms(),
    )
    val_ds = BratsSliceDataset(
        metadata_csv=METADATA_CSV_PATH,
        split="val",
        transform=build_val_transforms(),
    )

    print(f"train slices: {len(train_ds)}")
    print(f"val slices:   {len(val_ds)}")

    sample = train_ds[0]
    print("image shape:", sample["image"].shape, "label shape:", sample["label"].shape)
