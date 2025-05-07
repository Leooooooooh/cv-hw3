import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from src.dataset import InstanceSegDataset
from src.transforms import get_train_transform


def get_instance_segmentation_model(num_classes):
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights)

    # Replace classification head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask prediction head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def train(num_epochs=10, lr=1e-3, batch_size=2, model_save_dir="checkpoints"):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Dataset and DataLoader
    dataset = InstanceSegDataset("data", is_train=True, transforms=get_train_transform())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Make sure all labels are > 0
    print(f"Checking sample labels...")
    image, target = dataset[0]
    print(f"→ Sample labels: {target['labels']} (should be ≥ 1)")

    num_classes = 5  # background + 4 classes
    model = get_instance_segmentation_model(num_classes)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    os.makedirs(model_save_dir, exist_ok=True)
    loss_history = []

    model.train()
    for epoch in range(num_epochs):
        print(f"\n📘 Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = []

        for images, targets in tqdm(data_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            epoch_loss.append(losses.item())

        mean_loss = np.mean(epoch_loss)
        loss_history.append(mean_loss)
        print(f"📉 Avg Loss: {mean_loss:.4f}")

        # Save model
        save_path = os.path.join(model_save_dir, f"maskrcnn_epoch{epoch + 1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"💾 Saved model to {save_path}")

    # Plot loss history
    plt.plot(range(1, num_epochs + 1), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Avg Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    train()