# src/preprocessing/app.py
# python -m src.preprocessing.app

import logging
import subprocess  # noqa: F401
from pathlib import Path

import SimpleITK as sitk

from src.configs.config import BET_DATA_ROOT, PROCESSED_DIR, TRAIN_DATA_ROOT
from src.preprocessing.brain_extract import find_bet_t1c_path
from src.preprocessing.intensity_normalize import intensity_normalize
from src.preprocessing.npy_crop import is_already_processed, save_patient_slices
from src.preprocessing.resample import resample_image, resample_mask_to_reference

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SAVE_ROOT = PROCESSED_DIR / "patches"


def preprocess_image(image_path: Path):
    image = resample_image(image_path)
    image = intensity_normalize(image)
    return image


def preprocess_mask(mask_path: Path, processed_image: sitk.Image):
    label_mask = resample_mask_to_reference(mask_path, processed_image)
    return label_mask


def process_patient(patient_dir: Path, bet_root: Path, save_root: Path):
    t1c_path = next(patient_dir.glob("*_t1c.nii.gz"))
    gtv_path = next(patient_dir.glob("*_gtv.nii.gz"))
    # bet_t1c_path = find_bet_t1c_path(t1c_path, bet_root)

    img = preprocess_image(t1c_path)
    # img = preprocess_image(bet_t1c_path)
    mask = preprocess_mask(gtv_path, img)
    save_patient_slices(img, mask, patient_dir.name, save_root=save_root)


if __name__ == "__main__":
    patient_dirs = sorted(p for p in TRAIN_DATA_ROOT.iterdir() if p.is_dir())
    skipped, succeeded, no_bet, failed = [], [], [], []

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name

        if is_already_processed(patient_id, SAVE_ROOT):
            logger.info(f"Skip: [{patient_id}] 이미 처리됨")
            skipped.append(patient_id)
            continue

        try:
            process_patient(patient_dir, bet_root=BET_DATA_ROOT, save_root=SAVE_ROOT)
        except StopIteration:
            logger.error(f"Skip: [{patient_id}] t1c 또는 gtv 파일을 찾을 수 없음")
            failed.append(patient_id)
        # except subprocess.CalledProcessError as e:
        #     logger.error(f"Skip: [{patient_id}] HD-BET 실행 실패 (returncode={e.returncode})")
        #     failed.append(patient_id)
        except Exception:
            logger.exception(f"Skip: [{patient_id}] 처리 중 알 수 없는 오류 발생")
            failed.append(patient_id)
        else:
            succeeded.append(patient_id)

    logger.info(f"전체 {len(patient_dirs)}명 - 성공 {len(succeeded)}, 실패 {len(failed)}, 스킵 {len(skipped)}")
    if failed:
        logger.warning(f"실패한 환자 목록: {failed}")
