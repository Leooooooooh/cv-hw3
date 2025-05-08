import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from src.dataset import InstanceSegDataset
from src.model import get_instance_segmentation_model
import os
from tqdm import tqdm
from src.transforms import get_train_transform, get_val_transform
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

def collate_fn(batch):
    return tuple(zip(*batch))

def evaluate(model, data_loader, device):
    model.eval()
    coco_results = []
    image_ids = []
    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = [img.to(device) for img in images]
        with torch.no_grad():
            outputs = model(images)
        for output, target in zip(outputs, targets):
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()
            image_id = target.get("image_id", torch.tensor([0])).item()
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                coco_results.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                })
            image_ids.append(image_id)
    # Simulated COCOEval - place actual val annotations if available
    print(f"[Mock] Evaluated {len(image_ids)} samples. Save results to evaluate properly if needed.")

def train(num_epochs=16, lr=1e-4, batch_size=2, model_save_path="model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = InstanceSegDataset("data", is_train=True, transforms=get_train_transform())
    val_dataset = InstanceSegDataset("data", is_train=False, transforms=get_val_transform())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = get_instance_segmentation_model(num_classes=5)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    loss_history = []
    model.train()
    for epoch in range(num_epochs):
        print(f"\n📘 Epoch {epoch+1}/{num_epochs}")
        total_loss = 0.0

        for images, targets in tqdm(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        lr_scheduler.step()
        print(f"✅ Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), model_save_path)
        print(f"📦 Model saved to {model_save_path}")

        # Evaluate on val set
        print("🔎 Evaluating on validation set...")
        evaluate(model, val_loader, device)

    plt.plot(np.arange(1, num_epochs + 1), loss_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train()
