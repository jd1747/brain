# src/models & src/datasets/dataset.py 코드 통합본

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import rotate
from torch import nn
from torch.utils.data import Dataset

# 1. [dataset.py]
AUG_COMBINATIONS = [
    {"flip": None, "rotation": 0},
    {"flip": "horizontal", "rotation": 0},
    {"flip": None, "rotation": 90},
    {"flip": None, "rotation": -90},
    {"flip": "horizontal", "rotation": 90},
]

_NO_AUG = [{"flip": None, "rotation": 0}]


def _apply_augmentation(image: np.ndarray, mask: np.ndarray, combo: dict):
    if combo["flip"] == "horizontal":
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    elif combo["flip"] == "vertical":
        image = np.flipud(image)
        mask = np.flipud(mask)

    if combo["rotation"] != 0:
        image = np.rot90(image, k=1)
        mask = np.rot90(mask, k=1)

    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class BratsSliceDataset(Dataset):
    def __init__(self, metadata_csv: Path, split: str, augment: bool | None = None):
        assert split in {"train", "val", "test"}
        self.split = split

        if augment is None:
            augment = split == "train"
        if split != "train" and augment:
            raise ValueError("val/test data에는 증강 미적용")
        self.augment = augment

        df = pd.read_csv(metadata_csv)
        df = df[df["split"] == split]

        combos = AUG_COMBINATIONS if self.augment else _NO_AUG

        self.samples: list[tuple[Path, Path, dict]] = []
        for _, row in df.iterrows():
            image_dir = Path(row["image_dir"])
            mask_dir = Path(row["mask_dir"])

            for image_path in sorted(image_dir.glob("*.npy")):
                mask_path = mask_dir / image_path.name
                if not mask_path.exists():
                    continue
                for combo in combos:
                    self.samples.append((image_path, mask_path, combo))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, mask_path, combo = self.samples[idx]

        image = np.load(image_path)
        mask = np.load(mask_path)

        if combo["flip"] is not None or combo["rotation"] != 0:
            image, mask = _apply_augmentation(image, mask, combo)

        image = image[np.newaxis, ...].astype(np.float32)
        mask = mask[np.newaxis, ...].astype(np.float32)

        return image, mask


# 2. [model.py] 기본 UNet 구조
class DoubleConv(nn.Module):
    """(합성곱 -> 배치정규화 -> ReLU)를 2번 반복하는 U-Net의 기본 블록"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.boottleneck = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Downsampling path
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))

        b = self.boottleneck(self.pool4(d4))

        u1 = self.up1(b)
        u1 = torch.cat((u1, d4), dim=1)
        c1 = self.conv_up1(u1)

        u2 = self.up2(c1)
        u2 = torch.cat((u2, d3), dim=1)
        c2 = self.conv_up2(u2)

        u3 = self.up3(c2)
        u3 = torch.cat((u3, d2), dim=1)
        c3 = self.conv_up3(u3)

        u4 = self.up4(c3)
        u4 = torch.cat((u4, d1), dim=1)
        c4 = self.conv_up4(u4)

        return self.out(c4)


# 3. [train.py]
METADATA_CSV_PATH = Path("data/processed/metadata.csv")
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):
        bce_loss = self.bce(inputs, targets)
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return bce_loss + dice_loss


train_dataset = BratsSliceDataset(METADATA_CSV_PATH, split="train")
val_dataset = BratsSliceDataset(METADATA_CSV_PATH, split="val")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

model = UNet(in_channels=1, out_channels=1).to(DEVICE)
criterion = DiceBCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)


# --- 데이터 & 모델 준비 ---
print(f"\n🔥 총 {len(train_dataset)}장의 데이터로 뇌수막종 AI 학습을 시작합니다!")

for epoch in range(20):
    model.train()
    epoch_loss = 0.0

    # progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

    for images, masks in train_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        # progress_bar.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)
    print(f"🎯 Epoch {epoch + 1} 완료! 평균 Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "meningioma_unet.pth")
print("💾 학습 완료! 모델이 'meningioma_unet.pth' 로 안전하게 저장되었습니다.")
