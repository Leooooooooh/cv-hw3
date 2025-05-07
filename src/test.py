import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
from pycocotools import mask as mask_utils
from src.dataset import InstanceSegDataset
from src.model import get_instance_segmentation_model
from src.transforms import get_val_transform


def convert_to_coco_format(preds, image_ids):
    coco_results = []
    for pred, image_id in zip(preds, image_ids):
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        masks = pred["masks"].cpu().numpy()

        for box, score, label, mask in zip(boxes, scores, labels, masks):
            x1, y1, x2, y2 = box
            coco_box = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

            # Encode mask in RLE format (for COCO)
            rle = mask_utils.encode(np.asfortranarray(mask[0] > 0.5))
            rle["counts"] = rle["counts"].decode("utf-8")  # for JSON compatibility

            coco_results.append({
                "image_id": int(image_id),
                "category_id": int(label),
                "bbox": coco_box,
                "score": float(score),
                "segmentation": rle
            })
    return coco_results


def generate_predictions(model_path, val_data_root, output_path="predictions.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = get_instance_segmentation_model(num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Dataset
    val_dataset = InstanceSegDataset(root_dir=val_data_root, is_train=False, transforms=get_val_transform())
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    results = []
    image_ids = []

    for images, ids in tqdm(val_loader, desc="Generating predictions"):
        images = [img.to(device) for img in images]
        with torch.no_grad():
            outputs = model(images)

        results.extend(outputs)
        image_ids.extend([int(id) for id in ids])

    # Convert to COCO format
    coco_results = convert_to_coco_format(results, image_ids)

    # Write to JSON
    with open(output_path, "w") as f:
        json.dump(coco_results, f)
    print(f"✅ Saved {len(coco_results)} predictions to {output_path}")


if __name__ == "__main__":
    generate_predictions(
        model_path="model.pth",
        val_data_root="hw3-data-release",
        output_path="predictions.json"
    )