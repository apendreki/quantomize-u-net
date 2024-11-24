import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix

# ------------------------------
# 손실 함수 정의
# ------------------------------

def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

def combined_loss(pred, target):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    d_loss = dice_loss(pred, target)
    return bce + d_loss

# ------------------------------
# 데이터셋 클래스 정의
# ------------------------------

class DriveDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_list = sorted(os.listdir(image_dir))
        self.mask_list = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        else:
            transform_ops = transforms.Compose([
                transforms.Resize((576, 576)),
                transforms.ToTensor()
            ])
            image = transform_ops(image)
            mask = transform_ops(mask)

        mask = (mask > 0.5).float()
        return image, mask

# ------------------------------
# UNet 모델 정의
# ------------------------------

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class BasicUNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=1):
        super(BasicUNet, self).__init__()
        self.enc1 = UNetBlock(input_channels, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)
        self.bottleneck = UNetBlock(512, 1024)
        self.dec4 = UNetBlock(1024 + 512, 512)
        self.dec3 = UNetBlock(512 + 256, 256)
        self.dec2 = UNetBlock(256 + 128, 128)
        self.dec1 = UNetBlock(128 + 64, 64)
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(2)(enc2))
        enc4 = self.enc4(nn.MaxPool2d(2)(enc3))
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc4))

        dec4 = self.dec4(torch.cat([nn.functional.interpolate(bottleneck, scale_factor=2), enc4], dim=1))
        dec3 = self.dec3(torch.cat([nn.functional.interpolate(dec4, scale_factor=2), enc3], dim=1))
        dec2 = self.dec2(torch.cat([nn.functional.interpolate(dec3, scale_factor=2), enc2], dim=1))
        dec1 = self.dec1(torch.cat([nn.functional.interpolate(dec2, scale_factor=2), enc1], dim=1))

        return self.out_conv(dec1)

# ------------------------------
# 성능 계산 함수
# ------------------------------

def compute_metrics(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()

    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    accuracy = accuracy_score(labels, preds)

    # ROC AUC 계산 (예외 처리 추가)
    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = 0.0  # 모든 샘플이 한 클래스일 경우

    return precision, recall, f1, accuracy, auc

# ------------------------------
# 시각화 함수
# ------------------------------

def visualize_results(inputs, labels, preds, model_name, output_dir, idx):
    inputs = inputs.cpu().numpy().squeeze()
    labels = labels.cpu().numpy().squeeze()
    preds = preds.cpu().numpy().squeeze()

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(inputs, cmap='gray')
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(labels, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(preds, cmap='gray')
    plt.title(f'{model_name} Prediction')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_result_{idx}.png'))
    plt.close()

# ------------------------------
# 학습, 양자화, 테스트 함수
# ------------------------------

def train_model(model, train_loader, optimizer, num_epochs=10, device='cpu'):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = combined_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

def apply_static_quantization(model):
    return torch.quantization.quantize_dynamic(model, {nn.Conv2d}, dtype=torch.qint8)

def test_model(model, dataloader, device='cpu', model_name='Model', output_dir='results'):
    model.eval()
    all_preds = []
    all_labels = []

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = torch.sigmoid(model(inputs)) > 0.5
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # 시각화
            visualize_results(inputs[0], labels[0], outputs[0], model_name, output_dir, idx)

    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    return compute_metrics(all_preds, all_labels)

# ------------------------------
# 실행 코드
# ------------------------------

def main():
    train_image_dir = './data/DRIVE/training/images'
    train_mask_dir = './data/DRIVE/training/1st_manual'
    test_image_dir = './data/DRIVE/test/images'
    test_mask_dir = './data/DRIVE/test/1st_manual'

    transform = transforms.Compose([transforms.Resize((576, 576)), transforms.ToTensor()])
    train_dataset = DriveDataset(train_image_dir, train_mask_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataset = DriveDataset(test_image_dir, test_mask_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 모델 학습
    model = BasicUNet(input_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Training Original Model...")
    train_model(model, train_loader, optimizer, num_epochs=25, device=device)

    # 양자화
    print("Quantizing Model...")
    quantized_model = apply_static_quantization(model)

    # 테스트
    print("Testing Original Model...")
    original_metrics = test_model(model, test_loader, device=device, model_name='Original', output_dir='results/original')

    print("Testing Quantized Model...")
    quantized_metrics = test_model(quantized_model, test_loader, device=device, model_name='Quantized', output_dir='results/quantized')

    # 결과 비교
    print("\n=== Metrics Comparison ===")
    print("Original Model:")
    print(f"Precision: {original_metrics[0]:.4f}, Recall: {original_metrics[1]:.4f}, F1: {original_metrics[2]:.4f}, Accuracy: {original_metrics[3]:.4f}, AUC: {original_metrics[4]:.4f}")

    print("Quantized Model:")
    print(f"Precision: {quantized_metrics[0]:.4f}, Recall: {quantized_metrics[1]:.4f}, F1: {quantized_metrics[2]:.4f}, Accuracy: {quantized_metrics[3]:.4f}, AUC: {quantized_metrics[4]:.4f}")

if __name__ == "__main__":
    main()
