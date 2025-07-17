#!/usr/bin/env python3
"""
train_truck_bus.py

Nimmt beliebige Ordner mit vorab gesäuberten Bildern für die Klassen 'truck', 'bus'
und optional 'not_truck_bus', führt einen stratified 80/20 Split durch,
trainiert einen ResNet-50 Klassifikator und speichert das beste Modell.

Konfiguration:
  CLEANED_DIRS = {
    'truck': Path('path/to/cleaned_truck_folder'),
    'bus': Path('path/to/cleaned_bus_folder'),
    'not_truck_bus': Path('path/to/cleaned_negative_folder')
  }
  PRETRAINED_CHECK = Path('models/resnet50-0676ba61.pth')  # lokal gespeichertes ResNet-50
"""
import multiprocessing
from multiprocessing import freeze_support
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import shutil

# === KONFIGURATION ===
CLEANED_DIRS = {
    'truck':        Path(r'C:\Users\Erik\Desktop\Uni\HLCV\Project\data\highway\sorted\test\truck'),
    'bus':          Path(r'C:\Users\Erik\Desktop\Uni\HLCV\Project\data\highway\sorted\test\bus'),
    'not_truck_bus':Path(r'C:\Users\Erik\Desktop\Uni\HLCV\Project\data\highway\sorted\val\not_truck_or_bus')
}
# Globale Parameter
PRETRAINED_CHECK = Path(r'C:\Users\Erik\Desktop\Uni\HLCV\Project\models\resnet50-0676ba61.pth')
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 1e-4
DEVICE     = 'cuda:0' if torch.cuda.is_available() else 'cpu'
MODEL_OUT  = Path(r'C:\Users\Erik\Desktop\Uni\HLCV\Project\models\truck_bus_resnet50.pth')
NUM_WORKERS = 4
# ======================

def load_pretrained_resnet(path: Path, num_cls: int, device: str) -> nn.Module:
    model = models.resnet50(pretrained=False)
    state = torch.load(path, map_location='cpu')
    state.pop('fc.weight', None)
    state.pop('fc.bias', None)
    model.load_state_dict(state, strict=False)
    model.fc = nn.Linear(model.fc.in_features, num_cls)
    return model.to(device)


def main():
    # 1) Kombiniere alle Klassen in ein temporäres Verzeichnis
    ALL_DIR = Path(r'C:\Users\Erik\Desktop\Uni\HLCV\Project\data\highway\sorted\all_bus_truck_data')
    if ALL_DIR.exists():
        shutil.rmtree(ALL_DIR)
    for cls, folder in CLEANED_DIRS.items():
        dst = ALL_DIR / cls
        dst.mkdir(parents=True, exist_ok=True)
        for img in folder.glob('*'):
            if img.is_file() and img.suffix.lower() in ('.jpg','.jpeg','.png'):
                shutil.copy(img, dst / img.name)

    # 2) Daten-Transformationen
    tfm_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    tfm_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # 3) ImageFolder Dataset aus ALL_DIR
    full_dataset = datasets.ImageFolder(ALL_DIR, transform=tfm_train)
    labels = [full_dataset.samples[i][1] for i in range(len(full_dataset))]
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )
    train_ds = Subset(full_dataset, train_idx)
    val_folder = datasets.ImageFolder(ALL_DIR, transform=tfm_val)
    val_ds     = Subset(val_folder, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)

    # 4) Modell initialisieren und Pretrained laden
    num_classes = len(full_dataset.classes)
    model = load_pretrained_resnet(PRETRAINED_CHECK, num_classes, DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 5) Training
    best_acc = 0.0
    for epoch in range(1, EPOCHS+1):
        model.train()
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
        acc = correct / len(val_ds)
        print(f"Epoch {epoch}: Val Acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUT)

    print(f"Training beendet. Best Val Acc={best_acc:.4f}")

if __name__ == '__main__':
    freeze_support()
    main()
