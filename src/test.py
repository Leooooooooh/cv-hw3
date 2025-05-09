import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from pycocotools import mask as mask_utils

from src.dataset import InstanceSegDataset
from src.transforms import get_val_transform
from src.model import get_instance_segmentation_model

# Load test image ID mapping
with open("test_image_name_to_ids.json", "r") as f:
    image_id_data = json.load(f)
filename_to_id = {entry["file_name"]: entry["id"] for entry in image_id_data}

def convert_predictions_to_json(predictions, filenames, score_thresh=0.3):
    results = []
    for pred, fname in zip(predictions, filenames):
        image_id = filename_to_id.get(f"{fname}.tif")
        if image_id is None:
            print(f"⚠️ {fname}.tif not found in mapping, skipping.")
            continue

        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        masks = pred["masks"].cpu().numpy()
        height, width = masks.shape[-2:]

        sorted_indices = np.argsort(-scores)
        for idx in sorted_indices:
            score = scores[idx]
            if score < score_thresh:
                continue
            label = int(labels[idx])
            if label == 0:
                continue

            mask = (masks[idx][0] > 0.5).astype(np.uint8)
            rle = mask_utils.encode(np.asfortranarray(mask))
            rle["counts"] = rle["counts"].decode("utf-8")
            rle["size"] = [int(height), int(width)]

            x1, y1, x2, y2 = boxes[idx]
            bbox = [
                round(float(x1), 2),
                round(float(y1), 2),
                round(float(x2 - x1), 2),
                round(float(y2 - y1), 2),
            ]

            results.append({
                "image_id": image_id,
                "category_id": label,
                "bbox": bbox,
                "score": round(float(score), 4),
                "segmentation": rle
            })
    return results

def run_test(model_path, test_root="data", output_json="test-results.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_instance_segmentation_model(num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    test_dataset = InstanceSegDataset(root_dir=test_root, is_train=False, transforms=get_val_transform())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    all_preds = []
    all_filenames = []

    for images, ids in tqdm(test_loader, desc="🔍 Running Inference"):
        images = [img.to(device) for img in images]
        with torch.no_grad():
            outputs = model(images)

        all_preds.extend(outputs)
        all_filenames.extend(ids)

    result_json = convert_predictions_to_json(all_preds, all_filenames)

    with open(output_json, "w") as f:
        json.dump(result_json, f)

    print(f"✅ Saved {len(result_json)} predictions to {output_json}")

if __name__ == "__main__":
    run_test(
        model_path="model.pth",
        test_root="data",
        output_json="test-results.json"
    )