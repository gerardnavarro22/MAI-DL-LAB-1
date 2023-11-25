import os
import torch
import pandas as pd
import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from collections import Counter


class MAMEDataset(Dataset):
    """MAME dataset."""

    def __init__(self, labels_csv, images_dir, transform=None, target_transform=None, header=None):
        self.frame = pd.read_csv(labels_csv, header=header)
        self.root_dir = images_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.frame.iloc[idx, 0])
        # image = cv2.imread(img_name)
        image = Image.open(img_name).convert('RGB')
        label = self.frame.iloc[idx, 1]
        label = np.array([label], dtype=int)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.transform(label)

        return image, label

    def get_labels_distribution(self):
        labels = self.frame.iloc[:, 1]
        return dict(Counter(labels))

    def get_images(self):
        for idx in range(len(self.frame)):
            img_name = os.path.join(self.root_dir,
                                    self.frame.iloc[idx, 0])
            image = cv2.imread(img_name)
            yield image
