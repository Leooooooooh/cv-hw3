import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image

class InstanceSegDataset(Dataset):
    def __init__(self, root_dir, split='train', transforms=None):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        self.img_dir = os.path.join(root_dir, split)
        # assume each subfolder corresponds to one sample identified by an ID
        self.ids = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, img_id, 'image.tif')
        image = Image.open(img_path).convert("RGB")

        # load instance masks and labels
        masks, labels = [], []
        mask_dir = os.path.join(self.img_dir, img_id)
        for class_idx in range(1, 5):  # class1.tif to class4.tif
            mask_path = os.path.join(mask_dir, f'class{class_idx}.tif')
            if os.path.exists(mask_path):
                m = Image.open(mask_path)
                mask = F.to_tensor(m)[0] > 0
                masks.append(mask)
                labels.append(class_idx)

        if masks:
            masks = torch.stack(masks)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            masks = torch.zeros((0, *image.size[::-1]), dtype=torch.bool)
            labels = torch.tensor([], dtype=torch.int64)

        # compute bounding boxes for each mask
        boxes = []
        for mask in masks:
            pos = torch.where(mask)
            xmin, ymin = pos[1].min(), pos[0].min()
            xmax, ymax = pos[1].max(), pos[0].max()
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.tensor(boxes, dtype=torch.float32)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([idx])
        }

        # apply transforms
        if self.transforms:
            image, target = self.transforms(image, target)
        else:
            image = F.to_tensor(image)

        return image, target


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


def create_model(num_classes):
    # load a pretrained Mask R-CNN with ResNet50-FPN backbone
    model = maskrcnn_resnet50_fpn(pretrained=True)
    # replace classification head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # replace mask predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    model.train()
    for i, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if i % print_freq == 0:
            print(f"Epoch: {epoch} [{i}/{len(data_loader)}]  Loss: {losses.item():.4f}")


def evaluate(model, data_loader, device):
    model.eval()
    # TODO: implement test-time inference and metrics calculation
    pass


def main():
    root_dir = '/Users/wizzy/Documents/shcool/cv-dl/cv-hw3/hw3-data-release'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # total classes = 4 object classes + 1 background
    num_classes = 5
    # prepare datasets for training and testing
    dataset_train = InstanceSegDataset(root_dir, split='train', transforms=get_transform(train=True))
    dataset_test  = InstanceSegDataset(root_dir, split='test',  transforms=get_transform(train=False))

    # data loaders
    data_loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True,
                                   num_workers=4, collate_fn=collate_fn)
    data_loader_test  = DataLoader(dataset_test,  batch_size=2, shuffle=False,
                                   num_workers=4, collate_fn=collate_fn)

    # create and move model to device
    model = create_model(num_classes)
    model.to(device)

    # optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device)

    # save model checkpoint
    torch.save(model.state_dict(), 'maskrcnn_pretrained.pth')

if __name__ == '__main__':
    main()
