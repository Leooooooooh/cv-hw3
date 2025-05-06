import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image
import glob


class InstanceSegDataset(Dataset):
    def __init__(self, root_dir, is_train=True, transforms=None):
        self.root_dir = root_dir
        self.is_train = is_train
        self.transforms = transforms or ToTensor()
        self.samples = []

        if self.is_train:
            self.samples = sorted(os.listdir(os.path.join(root_dir, 'train')))
        else:
            self.samples = sorted(glob.glob(os.path.join(root_dir, 'test', '*.tif')))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_train:
            sample_id = self.samples[idx]
            img_path = os.path.join(self.root_dir, 'train', sample_id, 'image.tif')
            image = Image.open(img_path).convert("RGB")

            # Load all mask classes: class1.tif, class2.tif, ...
            mask_dir = os.path.join(self.root_dir, 'train', sample_id)
            masks = []
            labels = []
            for class_idx in range(1, 5):  # class1 to class4
                class_mask_path = os.path.join(mask_dir, f'class{class_idx}.tif')
                if not os.path.exists(class_mask_path):
                    continue
                class_mask = np.array(Image.open(class_mask_path))
                instance_ids = np.unique(class_mask)
                instance_ids = instance_ids[instance_ids > 0]

                for instance_id in instance_ids:
                    instance_mask = (class_mask == instance_id).astype(np.uint8)
                    masks.append(instance_mask)
                    labels.append(class_idx)

            if self.transforms:
                transformed = self.transforms(image=np.array(image), masks=masks)
                image = transformed["image"]
                masks = torch.stack([torch.tensor(m, dtype=torch.uint8) for m in transformed["masks"]])
            else:
                image = ToTensor()(image)
                masks = torch.stack([torch.tensor(m, dtype=torch.uint8) for m in masks])

            labels = torch.tensor(labels, dtype=torch.int64)
            target = {
                "masks": masks,
                "labels": labels,
            }
            return image, target

        else:
            img_path = self.samples[idx]
            image = Image.open(img_path).convert("RGB")
            image = self.transforms(image)
            image_id = os.path.basename(img_path).replace('.tif', '')
            return image, image_id