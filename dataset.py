import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class PartDrawingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, transform=None):
        self.noisy = sorted([os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir)])
        self.clean = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir)])
        self.transform = transform

    def __len__(self): return len(self.noisy)
    def __getitem__(self, idx):
      noisy = Image.open(self.noisy[idx]).convert("L")
      clean = Image.open(self.clean[idx]).convert("L")

      if self.transform:
        noisy = self.transform(noisy)
        clean = self.transform(clean)

      clean = (clean > 0.5).float()
      return noisy, clean

    
if __name__ == "__main__":
    data = seg_dataset(r'D:\Part_Drawing_2k.zip\dataset')
    print(data)
    print(len(data))
