import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.OneOf([
            A.ElasticTransform(p=0.3, alpha=1, sigma=50, alpha_affine=30),
            A.GridDistortion(p=0.3),
        ], p=0.4),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussianBlur(p=0.1),
        A.GaussNoise(p=0.1),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

def get_val_transform():
    return A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])