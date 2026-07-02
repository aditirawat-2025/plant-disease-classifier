"""
Plant Disease Classifier — Training Script
Transfer learning with ResNet18 on PlantVillage dataset.

Expected folder structure:
data/
  train/
    Tomato_Healthy/
    Tomato_Early_Blight/
    Tomato_Late_Blight/
    ...
  val/
    (same class subfolders)

Download dataset from Kaggle: "PlantVillage Dataset"
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import time
import copy
import random

# ---------- Config ----------
DATA_DIR = "data"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "plant_disease_model.pth"

# TRAIN_SUBSET=True uses a capped number of images per class instead of the full dataset.
# This keeps CPU training time reasonable (roughly 20-30 min) while still using enough
# data for a meaningful, real accuracy number for your resume project.
TRAIN_SUBSET = True
TRAIN_PER_CLASS = 250   # images per class for train
VAL_PER_CLASS = 60      # images per class for val
NUM_EPOCHS = 4

# ---------- Data transforms ----------
# Training gets augmentation to reduce overfitting; validation just gets resized/normalized
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

def main():
    # ---------- Datasets & loaders ----------
    image_datasets = {
        split: datasets.ImageFolder(f"{DATA_DIR}/{split}", data_transforms[split])
        for split in ["train", "val"]
    }

    class_names = image_datasets["train"].classes
    num_classes = len(class_names)

    if TRAIN_SUBSET:
        # Build a class-balanced subset so training uses a capped, manageable number of images
        random.seed(42)
        per_class_limit = {"train": TRAIN_PER_CLASS, "val": VAL_PER_CLASS}

        for split in ["train", "val"]:
            targets = image_datasets[split].targets  # class index for each image
            indices_by_class = {}
            for idx, label in enumerate(targets):
                indices_by_class.setdefault(label, []).append(idx)

            selected_indices = []
            for label, indices in indices_by_class.items():
                random.shuffle(indices)
                selected_indices.extend(indices[:per_class_limit[split]])

            image_datasets[split] = Subset(image_datasets[split], selected_indices)

        print(f"Using ~{TRAIN_PER_CLASS} train / {VAL_PER_CLASS} val images per class, {NUM_EPOCHS} epochs")

    dataloaders = {
        split: DataLoader(image_datasets[split], batch_size=BATCH_SIZE, shuffle=(split == "train"), num_workers=2)
        for split in ["train", "val"]
    }
    dataset_sizes = {split: len(image_datasets[split]) for split in ["train", "val"]}

    print(f"Classes found: {class_names}")
    print(f"Train size: {dataset_sizes['train']}, Val size: {dataset_sizes['val']}")
    print(f"Using device: {DEVICE}")

    # ---------- Model: transfer learning with ResNet18 ----------
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all layers except the final classifier — faster training, less overfitting on small data
    for param in model.parameters():
        param.requires_grad = False

    # Replace final fully-connected layer to match our number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    # Only optimize the new final layer's parameters
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    model = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=NUM_EPOCHS)

    # Save model weights + class names together for easy loading at inference time
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
    }, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

# ---------- Training loop ----------
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=10):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    start = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - start
    print(f"\nTraining complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    main()
