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

# Load the mapping from test_image_name_to_ids.json
with open("test_image_name_to_ids.json", "r") as f:
    image_id_data = json.load(f)
filename_to_id = {entry["file_name"]: entry["id"] for entry in image_id_data}

def convert_test_predictions(preds, filenames):
    results = []
    for pred, fname in zip(preds, filenames):
        image_id = filename_to_id.get(f"{fname}.tif")
        if image_id is None:
            raise ValueError(f"Image filename {fname} not found in mapping.")

        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        masks = pred["masks"].cpu().numpy()

        for box, score, label, mask in zip(boxes, scores, labels, masks):
            x1, y1, x2, y2 = box
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

            rle = mask_utils.encode(np.asfortranarray(mask[0] > 0.5))
            rle["counts"] = rle["counts"].decode("utf-8")

            results.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": bbox,
                "score": float(score),
                "segmentation": rle
            })
    return results


def predict_test(model_path, test_data_root, output_path="test_predictions.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_instance_segmentation_model(num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    test_dataset = InstanceSegDataset(root_dir=test_data_root, is_train=False, transforms=get_val_transform())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    predictions = []
    filenames = []

    for images, ids in tqdm(test_loader, desc="Predicting on test"):
        images = [img.to(device) for img in images]
        with torch.no_grad():
            outputs = model(images)

        predictions.extend(outputs)
        filenames.extend(ids)

    result_json = convert_test_predictions(predictions, filenames)

    with open(output_path, "w") as f:
        json.dump(result_json, f)
    print(f"✅ Saved {len(result_json)} predictions to: {output_path}")


if __name__ == "__main__":
    predict_test(
        model_path="model.pth",
        test_data_root="data",  # path to your test images
        output_path="test-results.json"
    )