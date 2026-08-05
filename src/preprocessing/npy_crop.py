from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.utils.utils import ProcessingParams


def center_crop_or_pad(image: np.ndarray, target_size=ProcessingParams.image_size):
    h, w = image.shape
    th, tw = target_size

    # Crop
    if h > th:
        top = (h - th) // 2
        image = image[top : top + th, :]
    if w > tw:
        left = (w - tw) // 2
        image = image[:, left : left + tw]

    # Pad
    h, w = image.shape
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)

    image = np.pad(
        image,
        (
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2),
        ),
        mode="constant",
    )

    return image


def save_patient_slices(
    image_sitk: sitk.Image,
    mask_sitk: sitk.Image,
    patient_id: str,
    save_root: Path,
    margin: int = 3,
):
    image = sitk.GetArrayFromImage(image_sitk)  # (z,y,x)
    mask = sitk.GetArrayFromImage(mask_sitk)

    if image.shape != mask.shape:
        raise ValueError(f"[{patient_id}] image/mask shape mismatch: image={image.shape}, mask={mask.shape}")

    patient_dir = save_root / patient_id
    image_dir = patient_dir / "image"
    mask_dir = patient_dir / "mask"

    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # 종양이 존재하는 slice -> 구분 없이 모조리 변환하는 걸로 변경
    positive = np.where(mask.reshape(mask.shape[0], -1).sum(axis=1) >= 0)[0]

    if len(positive) == 0:
        return

    start = max(0, positive.min() - margin)  # start=0, end=image.shape[0]해도 되는데
    end = min(image.shape[0], positive.max() + margin + 1)

    for z in range(start, end):
        img = center_crop_or_pad(image[z]).astype(np.float32)
        gt = center_crop_or_pad(mask[z]).astype(np.uint8)

        np.save(image_dir / f"{z:03d}.npy", img)
        np.save(mask_dir / f"{z:03d}.npy", gt)


def is_already_processed(patient_id: str, save_root: Path) -> bool:
    image_dir = save_root / patient_id / "image"
    mask_dir = save_root / patient_id / "mask"

    if not image_dir.exists() or not mask_dir.exists():
        return False

    return any(image_dir.glob("*.npy")) and any(mask_dir.glob("*.npy"))
