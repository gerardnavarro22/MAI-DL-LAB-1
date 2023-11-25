import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class OriginalPreprocess(Dataset):

    def __init__(self, subset, image_paths, image_labels, include_filename=False):
        self.subset = subset
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.include_filename = include_filename
        self.transform = transforms.Compose(self._get_transforms_list())
        self.label_transformation = None

        self.n_outputs = len(set(image_labels))
        self.set_up_label_transformation_for_classification()

    def set_up_label_transformation_for_classification(self):
        sorted_labels = sorted(set(self.image_labels))
        label2idx = {raw_label: idx for idx, raw_label in enumerate(sorted_labels)}
        self.label_transformation = lambda x: torch.tensor(label2idx[x], dtype=torch.long).view(-1, 1)

    def _get_transforms_list(self):
        return [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]

    def get_n_outputs(self):
        return self.n_outputs

    def __len__(self):
        return len(self.image_paths)

    def _custom_img_transformation(self, image):
        return image

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self._custom_img_transformation(image)

        label = self.image_labels[idx]

        if self.label_transformation:
            label = self.label_transformation(label)
        image = self.transform(image)

        if self.include_filename:
            return image, label, os.path.basename(img_path)
        else:
            return image, label


class DownsampledPreprocess(OriginalPreprocess):

    def __init__(self, *args, **kwargs):
        try:
            self.image_size = kwargs['size']
            del kwargs['size']
        except KeyError:
            self.image_size = (256, 256)
        self.crop_size = (int(self.image_size[0] * 0.875), int(self.image_size[1] * 0.875))
        super().__init__(*args, **kwargs)

    def _get_transforms_list(self):
        return_transform = [
            transforms.Resize(self.image_size),
            transforms.RandomRotation(degrees=30),
            transforms.RandomCrop(self.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.53116883, 0.57740572, 0.6089572], [0.26368123, 0.2632309, 0.26533898])]
        return return_transform
