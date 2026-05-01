"""
Bowling Segmentation - Training Script
Fine-tunes a pretrained Mask R-CNN on combined COCO-annotated bowling datasets.
Classes: ball, lane, pins
"""

import json
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision
from torchvision import transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_util
import cv2
import time
import matplotlib.pyplot as plt


CLASS_NAMES = {0: "background", 1: "ball", 2: "lane", 3: "pins"}


class BowlingDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation_file)
        # Filter to only jpg images (skip HEIC)
        self.ids = [
            img_id for img_id in sorted(self.coco.getImgIds())
            if self.coco.loadImgs(img_id)[0]['file_name'].lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        print(f"Dataset: {len(self.ids)} images (filtered out non-jpg)")

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        masks = []
        labels = []
        areas = []
        iscrowd = []

        w, h = img.size
        for ann in anns:
            # Get bbox in [x, y, w, h] format, convert to [x1, y1, x2, y2]
            x, y, bw, bh = ann['bbox']
            x1, y1, x2, y2 = float(x), float(y), float(x) + float(bw), float(y) + float(bh)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])  # 1=ball, 2=lane, 3=pins
            areas.append(float(ann['area']))
            iscrowd.append(ann.get('iscrowd', 0))

            # Create mask from segmentation (polygon or RLE)
            seg = ann['segmentation']
            if isinstance(seg, list):
                # Polygon format
                mask = np.zeros((h, w), dtype=np.uint8)
                for poly_pts in seg:
                    poly = np.array(poly_pts, dtype=np.float64).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [poly], 1)
            else:
                # RLE format
                if isinstance(seg['counts'], list):
                    rle = coco_mask_util.frPyObjects(seg, h, w)
                else:
                    rle = seg
                mask = coco_mask_util.decode(rle)
            masks.append(mask)

        if len(boxes) == 0:
            # No annotations - create empty targets
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "masks": torch.zeros((0, h, w), dtype=torch.uint8),
                "image_id": torch.tensor([img_id]),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros(0, dtype=torch.int64),
            }
        else:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "masks": torch.as_tensor(np.array(masks), dtype=torch.uint8),
                "image_id": torch.tensor([img_id]),
                "area": torch.as_tensor(areas, dtype=torch.float32),
                "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
            }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)


def get_transform():
    return T.Compose([T.ToTensor()])


def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_PATH = os.path.join(BASE_DIR, "bowling_maskrcnn_3class.pth")

    # Both dataset directories
    DATA_DIR_OLD = os.path.join(BASE_DIR, "bowling ball segment.coco-segmentation", "train")
    ANNO_OLD = os.path.join(DATA_DIR_OLD, "_annotations.coco.json")
    DATA_DIR_NEW = os.path.join(BASE_DIR, "Bowling.coco-segmentation (1)", "train")
    ANNO_NEW = os.path.join(DATA_DIR_NEW, "_annotations.coco.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load both datasets and combine
    tfms = get_transform()
    dataset_old = BowlingDataset(DATA_DIR_OLD, ANNO_OLD, transforms=tfms)
    dataset_new = BowlingDataset(DATA_DIR_NEW, ANNO_NEW, transforms=tfms)
    full_dataset = torch.utils.data.ConcatDataset([dataset_old, dataset_new])
    print(f"Combined dataset: {len(full_dataset)} images")

    # Split 85/15 train/val
    n = len(full_dataset)
    n_val = max(1, int(0.15 * n))
    n_train = n - n_val
    torch.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [n_train, n_val])
    print(f"Train: {n_train}, Val: {n_val}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    # 4 classes: background + ball + lane + pins
    model = get_model(num_classes=4)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    NUM_EPOCHS = 25
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            num_batches += 1

        lr_scheduler.step()
        avg_train_loss = epoch_loss / max(num_batches, 1)

        # Validate
        model.train()  # keep in train mode to get losses
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        elapsed = time.time() - t0

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Time: {elapsed:.1f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {SAVE_PATH}")

    # Plot train and validation loss curves
    plt.figure(figsize=(10, 6))
    epochs = range(1, NUM_EPOCHS + 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss', markersize=4)
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(BASE_DIR, "loss_curve.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Loss curve saved to: {plot_path}")


if __name__ == "__main__":
    main()
