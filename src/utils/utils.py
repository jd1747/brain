from dataclasses import dataclass


@dataclass
class RandomSeed:
    seed: int = 42


@dataclass
class HyperParams:
    lr: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 16
    epochs: int = 20
    threshold: float = 0.5


@dataclass
class ProcessingParams:
    image_size: tuple[int, int] = (256, 256)
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class DatasetParams:
    split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15)
