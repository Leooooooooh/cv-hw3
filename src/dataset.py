import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image
import glob
from tifffile import imread


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
            raw_masks = []
            raw_labels = []

            for class_idx in range(1, 5):  # class1 to class4
                class_mask_path = os.path.join(mask_dir, f'class{class_idx}.tif')
                if not os.path.exists(class_mask_path):
                    continue
                class_mask = imread(class_mask_path)
                instance_ids = np.unique(class_mask)
                instance_ids = instance_ids[instance_ids > 0]

                for instance_id in instance_ids:
                    instance_mask = (class_mask == instance_id).astype(np.uint8)
                    raw_masks.append(instance_mask)
                    raw_labels.append(class_idx)

            if self.transforms:
                transformed = self.transforms(image=np.array(image), masks=raw_masks)
                image = transformed["image"]
                transformed_masks = transformed["masks"]
            else:
                image = ToTensor()(image)
                transformed_masks = raw_masks

            # Process masks → tensor, remove empty masks
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

            if len(masks) == 0:
                # fallback: return dummy data to avoid crashing
                masks = torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8)
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
            else:
                masks = torch.stack(masks)
                boxes = torch.tensor(boxes, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.int64)

            target = {
                "masks": masks,
                "labels": labels,
                "boxes": boxes,
            }
            return image, target

        else:
            img_path = self.samples[idx]
            image = Image.open(img_path).convert("RGB")
            image = self.transforms(image)
            image_id = os.path.basename(img_path).replace('.tif', '')
            return image, image_id