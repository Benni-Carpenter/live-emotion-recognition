"""
Dataset classes for training and validation.
"""

from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.datasets as datasets

from src.config import IMAGE_SIZE


class LoadDataset(Dataset):
    def __init__(self, root):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.2), ratio=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.dataset = datasets.ImageFolder(root=root, transform=self.transform)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
