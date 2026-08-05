# src/preprocessing 코드 통합 (AI 질문용)

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.configs.config import IMAGE_SIZE, PROCESSED_DIR, TRAIN_DATA_ROOT


# 1. Resampling
def _read_image(file_path: Path) -> sitk.Image:
    return sitk.ReadImage(str(file_path))


def _orient_image(image: sitk.Image) -> sitk.Image:
    return sitk.DICOMOrient(image, "LPS")


def resample_image(
    image_path: Path, target_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear, default_px_value: float = 0.0
) -> sitk.Image:
    image = _read_image(image_path)
    image = _orient_image(image)
    old_spacing = image.GetSpacing()
    old_size = image.GetSize()

    new_size = [int(np.ceil(old_size[i] * old_spacing[i] / target_spacing[i])) for i in range(3)]

    resampler = sitk.ResampleImageFilter()

    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)

    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())

    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetDefaultPixelValue(default_px_value)

    resampler.SetInterpolator(interpolator)
    resampled = resampler.Execute(image)
    resampled = sitk.Cast(resampled, sitk.sitkFloat32)

    return resampled


def resample_mask_to_reference(mask_path: Path, reference_image: sitk.Image) -> sitk.Image:
    mask = _read_image(mask_path)
    mask = _orient_image(mask)

    return sitk.Resample(
        mask, reference_image, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkNearestNeighbor, 0, mask.GetPixelID()
    )


# 2. Brain Extraction (HD-BET)
def brain_extract(input_path: Path, output_path: Path):
    subprocess.run(
        [
            "hd-bet",
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "-device",
            "cuda",
        ],
        check=True,
    )


# 3. Intensity Normalization (z-score)
def intensity_normalize(image: sitk.Image) -> sitk.Image:
    arr = sitk.GetArrayFromImage(image).astype(np.float32)

    # 배경 제외
    mask = arr != 0

    if np.any(mask):
        mean = arr[mask].mean()
        std = arr[mask].std()
        arr[mask] = (arr[mask] - mean) / (std + 1e-8)

    image_norm = sitk.GetImageFromArray(arr)
    image_norm.CopyInformation(image)

    return image_norm


# 4. Crop/Padding and Save to .npy
def center_crop_or_pad(image: np.ndarray, target_size=IMAGE_SIZE):
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
    margin: int = 5,
):
    image = sitk.GetArrayFromImage(image_sitk)  # (z,y,x)
    mask = sitk.GetArrayFromImage(mask_sitk)

    patient_dir = save_root / patient_id
    image_dir = patient_dir / "image"
    mask_dir = patient_dir / "mask"

    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # 종양이 존재하는 slice
    positive = np.where(mask.reshape(mask.shape[0], -1).sum(axis=1) > 0)[0]

    if len(positive) == 0:
        return

    start = max(0, positive.min() - margin)
    end = min(image.shape[0], positive.max() + margin + 1)

    for z in range(start, end):
        img = center_crop_or_pad(image[z]).astype(np.float32)
        gt = center_crop_or_pad(mask[z]).astype(np.uint8)

        np.save(image_dir / f"{z:03d}.npy", img)
        np.save(mask_dir / f"{z:03d}.npy", gt)


# 5. Main Pipeline
SAVE_ROOT = PROCESSED_DIR / "patches"


def preprocess_image(image_path: Path):
    image = resample_image(image_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        bet_input_path = tmp_dir / "input.nii.gz"
        bet_output_path = tmp_dir / "output.nii.gz"

        sitk.WriteImage(image, str(bet_input_path))
        brain_extract(bet_input_path, bet_output_path)
        image = sitk.ReadImage(str(bet_output_path))
        image = sitk.Cast(image, sitk.sitkFloat32)

    image = intensity_normalize(image)
    return image


def preprocess_mask(mask_path: Path, processed_image: sitk.Image):
    label_mask = resample_mask_to_reference(mask_path, processed_image)
    return label_mask


if __name__ == "__main__":
    patient_dirs = sorted(p for p in TRAIN_DATA_ROOT.iterdir() if p.is_dir())

    for patient_dir in patient_dirs:
        t1c_path = next(patient_dir.glob("*_t1c.nii.gz"))
        gtv_path = next(patient_dir.glob("*_gtv.nii.gz"))

        img = preprocess_image(t1c_path)
        mask = preprocess_mask(gtv_path, img)
        save_patient_slices(img, mask, patient_dir.name, save_root=SAVE_ROOT)
