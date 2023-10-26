import os
import torch
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset
from collections import Counter

class MAMEDataset(Dataset):
    """MAME dataset."""

    def __init__(self, labels_csv, images_dir, transform=None, header=None):
        self.frame = pd.read_csv(labels_csv, header=header)
        self.root_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.frame.iloc[idx, 0])
        image = cv2.imread(img_name)
        label = self.frame.iloc[idx, 1]
        label = np.array([label], dtype=int)
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def get_labels_distribution(self):
        labels = self.frame.iloc[:, 1]
        return Counter(labels)
