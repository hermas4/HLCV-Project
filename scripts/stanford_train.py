#!/usr/bin/env python3
"""
stanford_train.py mit tqdm-Fortschrittsbalken
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm  # Fortschrittsbalken

# Konfiguration
DATA_DIR    = Path(r"C:\Users\Erik\Desktop\Uni\HLCV\Project\data\stanford\crops")
MODEL_PATH = Path(r"C:\Users\Erik\Desktop\Uni\HLCV\Project\models\resnet50-0676ba61.pth")
MODEL_OUT   = Path(r"C:\Users\Erik\Desktop\Uni\HLCV\Project\models\stanford_teacher_ep100v2.pth")
BATCH_SIZE  = 32
EPOCHS      = 100
LR          = 1e-3
NUM_WORKERS = 4
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"
PATIENCE    = 10

def load_pretrained_resnet(path: Path, n_classes: int, device: str) -> torch.nn.Module:
    # 1) ResNet-50 ohne pretrained-Flag, um lokalen Checkpoint zu laden
    model = models.resnet50(pretrained=False)
    # 2) Lade ImageNet-Weights (ohne FC)
    state = torch.load(path, map_location="cpu")
    state.pop('fc.weight', None)
    state.pop('fc.bias', None)
    model.load_state_dict(state, strict=False)
    # 3) Ersetze FC-Layer für Stanford-Klassen
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model.to(device)


def main():
    # 1) Datasets & DataLoader
    tfm_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    tfm_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=tfm_train)
    val_ds   = datasets.ImageFolder(DATA_DIR / "val",   transform=tfm_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)

    # 2) Modell initialisieren und Pretrained laden
    num_classes = len(train_ds.classes)
    model = load_pretrained_resnet(MODEL_PATH, num_classes, DEVICE)
    model = nn.DataParallel(model) if torch.cuda.device_count()>1 else model

    # 3) Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    epochs_no_improve = 0
    # 4) Trainings-Loop mit tqdm
    for epoch in range(1, EPOCHS+1):
        # Training
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", unit="batch")
        for imgs, labels in train_bar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*imgs.size(0)
            train_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_ds)
        # Validation
        model.eval()
        correct = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]  ", unit="batch")
        with torch.no_grad():
            for imgs, labels in val_bar:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs).argmax(dim=1)
                correct += (preds==labels).sum().item()
        epoch_acc = correct / len(val_ds)
        print(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Val Acc={epoch_acc:.4f}")

        # Best-Checkpoint
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUT)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping nach {epoch} Epochen ohne Verbesserung.")
                break

    print(f"\nTraining fertig. Best Val Acc: {best_acc:.4f}")
    print(f"Modell gespeichert in: {MODEL_OUT}")

if __name__ == "__main__":
    main()
