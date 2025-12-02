import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse
import wandb

from unet_model import UNet  # 替换为你的 UNet 模型文件

# =======================================================
# Dataset
# =======================================================
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, "input")
        self.target_dir = os.path.join(root_dir, "target")
        self.transform = transform
        self.files = sorted(os.listdir(self.input_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_input = Image.open(os.path.join(self.input_dir, fname)).convert("RGB")
        img_target = Image.open(os.path.join(self.target_dir, fname)).convert("RGB")

        if self.transform is not None:
            img_input = self.transform(img_input)
            img_target = self.transform(img_target)

        return img_input, img_target


# =======================================================
# Sobel 计算梯度幅值
# =======================================================
def compute_grad_mag(img):
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=img.device).view(1,1,3,3)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32, device=img.device).view(1,1,3,3)
    gray = img.mean(dim=1, keepdim=True)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
    return grad_mag


# =======================================================
# 空间自适应正则化（Weighted TV）
# =======================================================
class SpatialAdaptiveRegularization(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, img, weights):
        dx = img[:, :, :, 1:] - img[:, :, :, :-1]
        dy = img[:, :, 1:, :] - img[:, :, :-1, :]
        dx = F.pad(dx, (0,1,0,0))
        dy = F.pad(dy, (0,0,0,1))
        grad_mag = torch.sqrt(dx.pow(2).sum(dim=1, keepdim=True) +
                              dy.pow(2).sum(dim=1, keepdim=True) + self.eps)
        return (weights * grad_mag).mean()


# =======================================================
# Training Function
# =======================================================
def train(train_root, valid_root, batch_size=4, lr=1e-4, num_epochs=200, img_size=256, lambda_sarm=0.05):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # WandB
    wandb.init(
        project="InverseProject",
        entity="weikang-kong-kaust",
        config={
            "batch_size": batch_size,
            "lr": lr,
            "num_epochs": num_epochs,
            "img_size": img_size,
            "lambda_sarm": lambda_sarm
        }
    )

    # Transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    # Dataloader
    train_loader = DataLoader(
        ImageDataset(train_root, transform), batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    valid_loader = DataLoader(
        ImageDataset(valid_root, transform), batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # Models
    model_sarm = UNet(3,3).to(device)
    model_plain = UNet(3,3).to(device)

    criterion_l1 = nn.L1Loss()
    criterion_sarm = SpatialAdaptiveRegularization()
    optimizer_sarm = torch.optim.Adam(model_sarm.parameters(), lr=lr)
    optimizer_plain = torch.optim.Adam(model_plain.parameters(), lr=lr)

    # ===================================================
    # Training Loop
    # ===================================================
    for epoch in range(num_epochs):
        # ---- Training ----
        model_sarm.train()
        model_plain.train()

        sum_train_sarm = 0
        sum_train_plain = 0
        sum_edge_sarm = 0
        sum_edge_plain = 0
        sum_flat_sarm = 0
        sum_flat_plain = 0

        for lr_imgs, hr_imgs in train_loader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            # Edge mask
            grad_mag = compute_grad_mag(lr_imgs)
            threshold = torch.quantile(grad_mag, 0.9)
            edge_mask = (grad_mag > threshold).float()
            flat_mask = 1.0 - edge_mask

            # ===== SARM Model =====
            sr_sarm = model_sarm(lr_imgs)
            rec_loss = criterion_l1(sr_sarm, hr_imgs)
            weights = 1.0 / (grad_mag + 1e-3)
            weights = weights / weights.max()
            tv_loss = criterion_sarm(sr_sarm, weights)
            total_loss_sarm = rec_loss + lambda_sarm * tv_loss

            optimizer_sarm.zero_grad()
            total_loss_sarm.backward()
            optimizer_sarm.step()

            sum_train_sarm += total_loss_sarm.item()
            sum_edge_sarm += (edge_mask * torch.abs(sr_sarm - hr_imgs)).sum() / (edge_mask.sum() + 1e-6)
            sum_flat_sarm += (flat_mask * torch.abs(sr_sarm - hr_imgs)).sum() / (flat_mask.sum() + 1e-6)

            # ===== Plain Model =====
            sr_plain = model_plain(lr_imgs)
            plain_loss = criterion_l1(sr_plain, hr_imgs)
            optimizer_plain.zero_grad()
            plain_loss.backward()
            optimizer_plain.step()

            sum_train_plain += plain_loss.item()
            sum_edge_plain += (edge_mask * torch.abs(sr_plain - hr_imgs)).sum() / (edge_mask.sum() + 1e-6)
            sum_flat_plain += (flat_mask * torch.abs(sr_plain - hr_imgs)).sum() / (flat_mask.sum() + 1e-6)

        # Average training loss
        avg_train_sarm = sum_train_sarm / len(train_loader)
        avg_train_plain = sum_train_plain / len(train_loader)
        avg_edge_sarm = sum_edge_sarm / len(train_loader)
        avg_edge_plain = sum_edge_plain / len(train_loader)
        avg_flat_sarm = sum_flat_sarm / len(train_loader)
        avg_flat_plain = sum_flat_plain / len(train_loader)

        # ---- Validation ----
        model_sarm.eval()
        model_plain.eval()

        sum_valid_sarm = 0
        sum_valid_plain = 0
        sum_edge_valid_sarm = 0
        sum_edge_valid_plain = 0
        sum_flat_valid_sarm = 0
        sum_flat_valid_plain = 0

        with torch.no_grad():
            for lr_imgs, hr_imgs in valid_loader:
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

                grad_mag = compute_grad_mag(lr_imgs)
                threshold = torch.quantile(grad_mag, 0.9)
                edge_mask = (grad_mag > threshold).float()
                flat_mask = 1.0 - edge_mask

                # SARM
                sr_sarm = model_sarm(lr_imgs)
                rec_loss = criterion_l1(sr_sarm, hr_imgs)
                weights = 1.0 / (grad_mag + 1e-3)
                weights = weights / weights.max()
                tv_loss = criterion_sarm(sr_sarm, weights)
                total_loss_sarm = rec_loss + lambda_sarm * tv_loss
                sum_valid_sarm += total_loss_sarm.item()
                sum_edge_valid_sarm += (edge_mask * torch.abs(sr_sarm - hr_imgs)).sum() / (edge_mask.sum() + 1e-6)
                sum_flat_valid_sarm += (flat_mask * torch.abs(sr_sarm - hr_imgs)).sum() / (flat_mask.sum() + 1e-6)

                # Plain
                sr_plain = model_plain(lr_imgs)
                plain_loss = criterion_l1(sr_plain, hr_imgs)
                sum_valid_plain += plain_loss.item()
                sum_edge_valid_plain += (edge_mask * torch.abs(sr_plain - hr_imgs)).sum() / (edge_mask.sum() + 1e-6)
                sum_flat_valid_plain += (flat_mask * torch.abs(sr_plain - hr_imgs)).sum() / (flat_mask.sum() + 1e-6)

        # Average validation
        avg_valid_sarm = sum_valid_sarm / len(valid_loader)
        avg_valid_plain = sum_valid_plain / len(valid_loader)
        avg_edge_valid_sarm = sum_edge_valid_sarm / len(valid_loader)
        avg_edge_valid_plain = sum_edge_valid_plain / len(valid_loader)
        avg_flat_valid_sarm = sum_flat_valid_sarm / len(valid_loader)
        avg_flat_valid_plain = sum_flat_valid_plain / len(valid_loader)

        # Print
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train SARM: {avg_train_sarm:.4f} | Train Plain: {avg_train_plain:.4f} | "
              f"Edge SARM: {avg_edge_sarm:.4f} | Edge Plain: {avg_edge_plain:.4f} | "
              f"Valid SARM: {avg_valid_sarm:.4f} | Valid Plain: {avg_valid_plain:.4f} | "
              f"Edge Valid SARM: {avg_edge_valid_sarm:.4f} | Edge Valid Plain: {avg_edge_valid_plain:.4f}")

        # WandB log
        wandb.log({
            "epoch": epoch + 1,
            "train_loss_sarm": avg_train_sarm,
            "train_loss_plain": avg_train_plain,
            "valid_loss_sarm": avg_valid_sarm,
            "valid_loss_plain": avg_valid_plain,
            "edge_loss_sarm": avg_edge_sarm,
            "edge_loss_plain": avg_edge_plain,
            "flat_loss_sarm": avg_flat_sarm,
            "flat_loss_plain": avg_flat_plain,
            "valid_edge_sarm": avg_edge_valid_sarm,
            "valid_edge_plain": avg_edge_valid_plain,
            "valid_flat_sarm": avg_flat_valid_sarm,
            "valid_flat_plain": avg_flat_valid_plain
        })

    # Save final models
    torch.save(model_sarm.state_dict(), "unet_sarm_final.pth")
    torch.save(model_plain.state_dict(), "unet_plain_final.pth")
    print("Training finished! Models saved.")


# =======================================================
# Main
# =======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--valid_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--lambda_sarm", type=float, default=0.05)
    args = parser.parse_args()

    train(
        args.train_root,
        args.valid_root,
        args.batch_size,
        args.lr,
        args.num_epochs,
        args.img_size,
        args.lambda_sarm
    )


