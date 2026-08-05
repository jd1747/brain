# python -m src.evaluate.evaluate

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch import nn

from src.configs.config import METADATA_CSV_PATH, OUTPUT_ROOT
from src.models.model import UNet
from src.utils.utils import HyperParams, ProcessingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


# 1. 지표 계산
def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    # return float((2.0 * intersection + eps) / (pred.sum() + gt.sum() + eps))
    return float((intersection + eps) / (gt.sum() + eps))


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    eroded = binary_erosion(mask)
    surface = mask & ~eroded
    return np.array(np.nonzero(surface)).T


def hausdorff_distance_95(pred: np.ndarray, gt: np.ndarray, spacing=ProcessingParams.spacing) -> float:
    """
    95% Hausdorff Distance (mm)
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")

    pred_surface = _surface_voxels(pred)
    gt_surface = _surface_voxels(gt)

    gt_dist_map = distance_transform_edt(~gt, sampling=spacing)
    pred_dist_map = distance_transform_edt(~pred, sampling=spacing)

    d_pred_to_gt = gt_dist_map[tuple(pred_surface.T)]  # type: ignore
    d_gt_to_pred = pred_dist_map[tuple(gt_surface.T)]  # type: ignore

    return float(np.percentile(np.concatenate([d_pred_to_gt, d_gt_to_pred]), 95))


# 2. 환자 단위 volume 로드 / 추론
def load_patient_volume(image_dir: Path, mask_dir: Path):
    image_paths = sorted(image_dir.glob("*.npy"))
    if not image_paths:
        raise FileNotFoundError(f"slice 없음: {image_dir}")

    images, masks = [], []
    for image_path in image_paths:
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"대응하는 mask 없음: {mask_path}")
        images.append(np.load(image_path))
        masks.append(np.load(mask_path))

    image_volume = np.stack(images, axis=0).astype(np.float32)  # (Z, H, W)
    mask_volume = np.stack(masks, axis=0).astype(np.uint8)
    return image_volume, mask_volume


@torch.no_grad()
def predict_volume(model: nn.Module, image_volume: np.ndarray, batch_size: int = 8) -> np.ndarray:
    model.eval()
    z = image_volume.shape[0]
    pred_slices = []

    for start in range(0, z, batch_size):
        batch = image_volume[start : start + batch_size]
        batch_tensor = torch.from_numpy(batch).unsqueeze(1).to(DEVICE)  # (b, 1, H, W)

        logits = model(batch_tensor)
        probs = torch.sigmoid(logits)
        pred = (probs > HyperParams.threshold).squeeze(1).cpu().numpy().astype(np.uint8)
        pred_slices.append(pred)

    return np.concatenate(pred_slices, axis=0)


# 3. 메인 평가 루프
def evaluate(metadata_csv: Path, checkpoint_path: Path, split: str = "test") -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    df = df[df["split"] == split]

    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    results = []
    for _, row in df.iterrows():
        patient_id = row["patient_id"]
        image_dir = Path(row["image_dir"])
        mask_dir = Path(row["mask_dir"])

        try:
            image_volume, mask_volume = load_patient_volume(image_dir, mask_dir)
        except FileNotFoundError as e:
            logger.warning(f"Skip: [{patient_id}] {e}")
            continue

        pred_volume = predict_volume(model, image_volume)

        dsc = dice_score(pred_volume, mask_volume)
        hd95 = hausdorff_distance_95(pred_volume, mask_volume)

        logger.info(f"[{patient_id}] DSC={dsc:.4f}, 95HD={hd95:.2f}mm")
        results.append({"patient_id": patient_id, "dsc": dsc, "hd95": hd95})

    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("meningioma_unet.pth"))
    parser.add_argument("--out_csv", type=Path, default=f"{OUTPUT_ROOT / 'eval_results.csv'}")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    result_df = evaluate(METADATA_CSV_PATH, args.checkpoint, split="test")
    result_df.to_csv(args.out_csv, index=False)

    valid_hd95 = result_df["hd95"].dropna()
    logger.info(
        f"평균 DSC: {result_df['dsc'].mean():.4f} (\u00b1{result_df['dsc'].std():.4f}), "
        f"평균 95HD: {valid_hd95.mean():2f}mm (\u00b1{valid_hd95.std():.2f}mm, "
        f"nan 제외 {len(valid_hd95)}/{len(result_df)}명)"
    )
    logger.info(f"환자별 결과 저장 완료: {args.out_csv}")
