# src/configs/config.py

from pathlib import Path

# ────────────────────────────────────────────────────────────────
# 프로젝트 경로
# 현재 파일: PROJECT_ROOT/src/configs/config.py
# ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_CSV_PATH = PROCESSED_DIR / "metadata.csv"

# ────────────────────────────────────────────────────────────────
# 데이터 경로 (읽기 전용)
# ────────────────────────────────────────────────────────────────
SERVER_ROOT = Path("/home/ljy/brain/data")
RAW_ROOT = SERVER_ROOT / "raw"
TRAIN_DATA_ROOT = RAW_ROOT / "BraTS-MEN-RT-Train-v2"
BET_DATA_ROOT = RAW_ROOT / "t1_bet"

# ────────────────────────────────────────────────────────────────
# 결과 출력 경로
# ────────────────────────────────────────────────────────────────
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CKPT_DIR = OUTPUT_ROOT / "checkpoints"
LOG_DIR = OUTPUT_ROOT / "logs"
FIGURE_DIR = OUTPUT_ROOT / "figures"
GRADCAM_DIR = OUTPUT_ROOT / "gradcam"

# ────────────────────────────────────────────────────────────────
# 디렉토리 자동 생성 ('데이터 경로' 제외)
# ────────────────────────────────────────────────────────────────
DIRS_TO_CREATE = [DATA_DIR, PROCESSED_DIR, OUTPUT_ROOT, CKPT_DIR, LOG_DIR, FIGURE_DIR, GRADCAM_DIR]
for path in DIRS_TO_CREATE:
    path.mkdir(parents=True, exist_ok=True)
