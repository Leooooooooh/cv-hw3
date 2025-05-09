import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image
from tifffile import imread
from sklearn.model_selection import train_test_split


class InstanceSegDataset(Dataset):
    def __init__(self, root_dir, is_train=True, transforms=None, val_split=0.1, seed=42):
        self.root_dir = root_dir
        self.is_train = is_train
        self.transforms = transforms or ToTensor()

        all_samples = sorted(os.listdir(os.path.join(root_dir, 'train')))
        train_samples, val_samples = train_test_split(
            all_samples, test_size=val_split, random_state=seed
        )
        self.samples = train_samples if is_train else val_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        img_path = os.path.join(self.root_dir, 'train', sample_id, 'image.tif')
        image = Image.open(img_path).convert("RGB")

        # Load all instance masks from class1.tif to class4.tif
        mask_dir = os.path.join(self.root_dir, 'train', sample_id)
        raw_masks = []
        raw_labels = []

        for class_idx in range(1, 5):  # class1 to class4
            class_mask_path = os.path.join(mask_dir, f'class{class_idx}.tif')
            if not os.path.exists(class_mask_path):
                continue
            class_mask = imread(class_mask_path)
            instance_ids = np.unique(class_mask)
            instance_ids = instance_ids[instance_ids > 0]  # ignore background

            for instance_id in instance_ids:
                binary_mask = (class_mask == instance_id).astype(np.uint8)
                raw_masks.append(binary_mask)
                raw_labels.append(class_idx)

        if self.transforms:
            transformed = self.transforms(image=np.array(image), masks=raw_masks)
            image = transformed["image"]
            transformed_masks = transformed["masks"]
        else:
            image = ToTensor()(image)
            transformed_masks = raw_masks

        # Final cleanup
        masks = []
        labels = []
        boxes = []
        for i, m in enumerate(transformed_masks):
            m_tensor = torch.tensor(m, dtype=torch.uint8)
            pos = torch.nonzero(m_tensor)
            if pos.numel() == 0:
                continue
            xmin, ymin = pos.min(dim=0)[0]
            xmax, ymax = pos.max(dim=0)[0]
            boxes.append([xmin.item(), ymin.item(), xmax.item(), ymax.item()])
            masks.append(m_tensor)
            labels.append(raw_labels[i])


        masks = torch.stack(masks)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "masks": masks,
            "labels": labels,
            "boxes": boxes,
        }
        return image, target
