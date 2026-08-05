# python -m src.models.train

from datetime import datetime

import torch
from torch import nn
from tqdm import tqdm

from src.configs.config import METADATA_CSV_PATH
from src.datasets.dataset import BratsSliceDataset
from src.models.model import UNet
from src.utils.utils import HyperParams

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


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

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=HyperParams.batch_size, shuffle=True, num_workers=8
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=HyperParams.batch_size, shuffle=False, num_workers=8)

model = UNet(in_channels=1, out_channels=1).to(DEVICE)
criterion = DiceBCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=HyperParams.lr, weight_decay=HyperParams.weight_decay)


# --- 데이터 & 모델 준비 ---
print(f"\n🔥 총 {len(train_dataset)}장의 데이터로 뇌수막종 AI 학습을 시작합니다!")

for epoch in range(HyperParams.epochs):
    model.train()
    epoch_loss = 0.0
    total = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{HyperParams.epochs}")

    for images, masks in progress_bar:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * masks.shape[0]
        total += masks.shape[0]
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / total
    print(f"🎯 Epoch {epoch + 1} - Train Set - 평균 Loss: {avg_loss:.4f}")

    with torch.no_grad():
        model.eval()
        val_loss = 0.0
        total = 0

        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item() * masks.shape[0]
            total += masks.shape[0]

        avg_loss = val_loss / total
        print(f"Epoch {epoch + 1} - Validation Set - 평균 Loss: {avg_loss:.4f}")


torch.save(model.state_dict(), "meningioma_unet.pth")
print("💾 학습 완료! 모델이 'meningioma_unet.pth' 로 안전하게 저장되었습니다.")
