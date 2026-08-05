# src/preprocessing/resample.py

from pathlib import Path

import numpy as np
import SimpleITK as sitk


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
