import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from src.dataset import InstanceSegDataset
from src.model import get_instance_segmentation_model
import os
from tqdm import tqdm
from src.transforms import get_train_transform

def collate_fn(batch):
    return tuple(zip(*batch))

def train(num_epochs=10, lr=1e-4, batch_size=2, model_save_path="model.pth"):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    dataset = InstanceSegDataset("data", is_train=True, transforms=get_train_transform())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Model
    model = get_instance_segmentation_model(num_classes=5)
    model.to(device)

    # Optimizer and LR scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0.0

        for images, targets in tqdm(data_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        lr_scheduler.step()
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

        # Save model
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    train()