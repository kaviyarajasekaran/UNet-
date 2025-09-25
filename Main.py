import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import os,torch, torch.nn as nn, torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from zipfile import ZipFile
from unet import unet
import torch.nn.functional as F
from dataset import PartDrawingDataset
from Loss_function import FocalLoss

if __name__ == "__main__":
    lr = 1e-4
    batch_size = 8
    EPOCHS = 30
    root = "Part_Drawing_2k/dataset"
    device = 'cuda' if t.cuda.is_available() else 'cpu'
    
train_transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
])

full_dataset = PartDrawingDataset(
    os.path.join(root,"X"),
    os.path.join(root,"Y"),
    transform=None
)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_set, val_set = random_split(full_dataset, [train_size, val_size])

train_set.dataset.transform = train_transform
val_set.dataset.transform = val_transform

train_loader = DataLoader(train_set, batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size)

model = unet( in_channel=1, num_classes=1).to(device)
criterion = FocalLoss(alfa=0.75, gamma=2)
optimizer = optim.Adam(model.parameters(), lr)

scaler = torch.amp.GradScaler()
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for noisy, clean in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        noisy = noisy.to(device)
        clean = clean.to(device).float()

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            outputs = model(noisy)
            loss = criterion(outputs, clean)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
    print(f"Epoch {epoch+1}, Train Loss={train_loss/len(train_loader):.4f}")

torch.save(model.state_dict(),"denoise_cbam.pth")
def show_results(dataset, model, device, num_batches=3):
    model.eval()

    train_iter = iter(train_loader)

    for _ in range(num_batches):
        try:
            noisy, clean = next(train_iter)
        except StopIteration:
            print("Restarting DataLoader iterator...")
            train_iter = iter(train_loader)
            noisy, clean = next(train_iter)

        noisy = noisy.to(device)
        clean = clean.to(device)

        with torch.no_grad():
            output = model(noisy).cpu()

        binarized_output = (output > 0.5).float()

        for i in range(len(noisy)):
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

            axs[0].imshow(noisy[i][0].cpu(), cmap="gray")
            axs[0].set_title("Noisy Input")
            axs[0].axis("off")

            axs[1].imshow(binarized_output[i][0], cmap="gray")
            axs[1].set_title("Denoised Output")
            axs[1].axis("off")

            axs[2].imshow(clean[i][0].cpu(), cmap="gray")
            axs[2].set_title("Ground Truth")
            axs[2].axis("off")

            plt.show()
show_results(val_set, model, device, num_batches=5)
