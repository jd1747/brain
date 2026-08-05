# notebook/size_check.py
"""
BraTS 데이터 전처리 사전 점검 스크립트.

* 실행: python -m notebook.size_check

1. 원본 데이터(Train_v2) 내 t1c-gtv 공간정보 일치 여부 확인
  - 공간정보: size, spacing, origin, direction
  - 시행 결과 해당 메타데이터는 완벽히 일치함

2. brain extraction 처리된 데이터(t1_bet)와 환자 id 매칭 & 공간정보 일치 여부 확인
  - t1_bet에 누락된 항목: ['BraTS-MEN-RT-0402-1']
  - 저거 빼고 나머지 499개의 경우 direction 값 오차가 나온 경우(29개)는 있으나, 경미한 수준. (1e-6 미만)

3. t1_bet 폴더 내 t1c 파일들 size/spacing/physical_size 분포 확인 및 축(xyz)별 시각화
  - physical_size: 실제 물리적 거리(mm). [size * spacing]
  - 당장 모델 돌리는 건 t1_bet 데이터 쓸 것 같아서 이걸로 설정했음. (변경될 수 있음)
"""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from src.configs.config import BET_DATA_ROOT, FIGURE_DIR, TRAIN_DATA_ROOT

AXIS_LABELS = ("x", "y", "z")


# ─────────────────────────────────────────────────────────────────
# 1. 공통 유틸
# ─────────────────────────────────────────────────────────────────
def _read_meta(path: Path) -> dict:
    """
    nii.gz 파일에서 메타정보만 읽음.
    - sitk.ReadImage() 사용 시 이미지 전체를 읽기 때문에 속도 느려짐
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()

    return {
        "size": reader.GetSize(),
        "spacing": reader.GetSpacing(),
        "origin": reader.GetOrigin(),
        "direction": reader.GetDirection(),
    }


def _meta_equal(meta_a: dict, meta_b: dict, keys=("size", "spacing", "origin", "direction")) -> bool:
    return all(meta_a[k] == meta_b[k] for k in keys)


# ─────────────────────────────────────────────────────────────────
# 2. Train_v2 내 t1c-gtv 페어 간 공간정보 일치 여부 확인
# ─────────────────────────────────────────────────────────────────
def check_gtv_t1c_consistency(train_root: Path) -> dict:
    """환자별 t1c-gtv 메타정보 일치 여부 확인"""
    patient_dirs = sorted(p for p in train_root.iterdir() if p.is_dir())
    mismatch = []

    for patient_dir in patient_dirs:
        gtv_path = next(patient_dir.glob("*_gtv.nii.gz"))
        t1c_path = next(patient_dir.glob("*_t1c.nii.gz"))

        gtv_meta = _read_meta(gtv_path)
        t1c_meta = _read_meta(t1c_path)

        if not _meta_equal(gtv_meta, t1c_meta):
            mismatch.append({"patient": patient_dir.name, "gtv": gtv_meta, "t1c": t1c_meta})

    return {"total": len(patient_dirs), "mismatch": mismatch}


def print_gtv_t1c_report(result: dict, max_print: int = 5) -> None:
    print(f"총 환자 수: {result['total']}")
    print(f"Mismatch: {len(result['mismatch'])}")

    for m in result["mismatch"][:max_print]:
        print(m["patient"])
        print(f"- Size: {(m['gtv']['size'], m['t1c']['size'])}")
        print(f"- Spacing: {(m['gtv']['spacing'], m['t1c']['spacing'])}")
        print(f"- Origin: {(m['gtv']['origin'], m['t1c']['origin'])}")
        print(f"- Direction: {(m['gtv']['direction'], m['t1c']['direction'])}")


# ─────────────────────────────────────────────────────────────────
# 3. Train_v2 - t1_bet 간 환자 id / t1c 공간정보 일치 여부 확인
# ─────────────────────────────────────────────────────────────────
def check_train_bet_consistency(train_root: Path, bet_root: Path) -> dict:
    """
    train - bet 환자 id 매칭 & t1c 메타정보 일치 여부 확인.

    direction에서 오차가 발생한 경우가 있으나, 1e-7 미만, 부동소수점 오차 수준이라 비교 대상에서 제외함.
    - spacing, size, origin 값은 전부 일치함
    """
    train_ids = {p.name for p in train_root.iterdir() if p.is_dir()}
    bet_ids = {f.name.replace("_t1c.nii.gz", "") for f in bet_root.glob("*.nii.gz")}
    common_ids = train_ids & bet_ids

    mismatch = []
    for pid in sorted(common_ids):
        train_path = next((train_root / pid).glob("*_t1c.nii.gz"))
        bet_path = bet_root / f"{pid}_t1c.nii.gz"

        train_meta = _read_meta(train_path)
        bet_meta = _read_meta(bet_path)

        if not _meta_equal(train_meta, bet_meta, keys=("size", "spacing", "origin")):
            mismatch.append({"patient": pid, "train": train_meta, "bet": bet_meta})

    return {
        "train_total": len(train_ids),
        "bet_total": len(bet_ids),
        "train_only": sorted(train_ids - bet_ids),
        "bet_only": sorted(bet_ids - train_ids),
        "common_ids": common_ids,
        "mismatch": mismatch,
    }


def print_train_bet_report(result: dict) -> None:
    print(f"Train: {result['train_total']}")
    print(f"BET: {result['bet_total']}")

    print(f"\nBET에 없는 환자: {result['train_only']}")
    print(f"\nBET에만 있는 환자: {result['bet_only']}")

    print(f"\nMismatch: {len(result['mismatch'])}")


# ─────────────────────────────────────────────────────────────────
# 4. size/spacing/physical_size 분포 (t1_bet 데이터 기준)
# ─────────────────────────────────────────────────────────────────
def collect_spatial_distribution(bet_root: Path, patient_ids: set) -> dict:
    spacing_list, size_list, direction_list, physical_list = [], [], [], []

    for pid in sorted(patient_ids):
        meta = _read_meta(bet_root / f"{pid}_t1c.nii.gz")
        spacing = meta["spacing"]
        size = meta["size"]

        spacing_list.append(spacing)
        size_list.append(size)
        direction_list.append(meta["direction"])
        physical_list.append(tuple(np.array(spacing) * np.array(size)))

    return {
        "spacing": spacing_list,
        "size": size_list,
        "direction": direction_list,
        "physical_size": physical_list,
    }


def print_distribution_summary(dist: dict) -> None:
    for key in ("size", "spacing", "physical_size", "direction"):
        most_common, count = Counter(dist[key]).most_common(1)[0]
        print(f"{key}: {most_common} ({count}건)")


def plot_size_distribution(sizes: np.ndarray, z_bins: int = 50):
    """size는 정수 voxel 수.

    x,y는 한 도표에 담되 값을 합치지 않고 x/y 막대를 나란히(그룹 막대) 표시해서
    구분 가능하게 함. z는 값이 다양하게 흩어져 있어 구간(bin)을 나눈 히스토그램으로 표시.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x_counter = Counter(sizes[:, 0].tolist())
    y_counter = Counter(sizes[:, 1].tolist())
    keys = sorted(set(x_counter) | set(y_counter))
    idx = np.arange(len(keys))
    width = 0.2

    axes[0].bar(idx - width / 2, [x_counter.get(k, 0) for k in keys], width, label="x")
    axes[0].bar(idx + width / 2, [y_counter.get(k, 0) for k in keys], width, label="y")
    axes[0].set_xticks(idx)
    axes[0].set_xticklabels([str(k) for k in keys])
    axes[0].set_title("size - x,y")
    axes[0].set_xlabel("voxels")
    axes[0].set_ylabel("Patients")
    axes[0].legend()

    axes[1].hist(sizes[:, 2], bins=z_bins)
    axes[1].set_title("size - z")
    axes[1].set_xlabel("voxels")
    axes[1].set_ylabel("Patients")

    fig.suptitle("Size distribution")
    fig.tight_layout()
    return fig


def plot_continuous_distribution(values: np.ndarray, metric_name: str, bins: int = 50):
    """spacing, physical_size처럼 연속값을 갖는 항목은 히스토그램으로 표시.

    x,y는 한 도표에 담되 값을 합치지 않고 x/y 히스토그램을 나란히(범례로 구분) 표시.
    z는 별도 도표로 표시.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist([values[:, 0], values[:, 1]], bins=bins, label=["x", "y"])
    axes[0].set_title(f"{metric_name} - x,y")
    axes[0].set_xlabel(metric_name)
    axes[0].set_ylabel("Patients")
    axes[0].legend()

    axes[1].hist(values[:, 2], bins=bins)
    axes[1].set_title(f"{metric_name} - z")
    axes[1].set_xlabel(metric_name)
    axes[1].set_ylabel("Patients")

    fig.suptitle(f"{metric_name} distribution")
    fig.tight_layout()
    return fig


def plot_distribution(dist: dict, save_dir: Path | None = None) -> None:
    size_arr = np.array(dist["size"])
    spacing_arr = np.array(dist["spacing"])
    physical_arr = np.array(dist["physical_size"])

    fig_size = plot_size_distribution(size_arr)
    fig_spacing = plot_continuous_distribution(spacing_arr, "spacing")
    fig_physical = plot_continuous_distribution(physical_arr, "physical_size")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig_size.savefig(save_dir / "size_distribution.png")
        fig_spacing.savefig(save_dir / "spacing_distribution.png")
        fig_physical.savefig(save_dir / "physical_size_distribution.png")
        plt.close("all")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────
# 5. 실행
# ─────────────────────────────────────────────────────────────────
def process():
    gtv_result = check_gtv_t1c_consistency(TRAIN_DATA_ROOT)
    print_gtv_t1c_report(gtv_result)

    print()
    train_bet_result = check_train_bet_consistency(TRAIN_DATA_ROOT, BET_DATA_ROOT)
    print_train_bet_report(train_bet_result)

    print()
    dist = collect_spatial_distribution(BET_DATA_ROOT, train_bet_result["common_ids"])
    print_distribution_summary(dist)
    plot_distribution(dist, save_dir=FIGURE_DIR)


if __name__ == "__main__":
    process()
