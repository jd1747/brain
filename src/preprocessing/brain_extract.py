import subprocess
from pathlib import Path


def brain_extract(input_path: Path, output_path: Path):
    subprocess.run(
        [
            "hd-bet",
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "-device",
            "cuda:1",
        ],
        check=True,
    )


def find_bet_t1c_path(t1c_path: Path, bet_root: Path) -> Path:
    bet_t1c_path = bet_root / t1c_path.name
    if not bet_t1c_path.exists():
        raise FileNotFoundError(f"BET 데이터 없음: {bet_t1c_path}")
    return bet_t1c_path
