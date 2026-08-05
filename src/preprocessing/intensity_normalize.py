import numpy as np
import SimpleITK as sitk


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
