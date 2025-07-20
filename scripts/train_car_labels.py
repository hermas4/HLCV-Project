#!/usr/bin/env python3
"""
train_car_classifier.py

Trainiert ein CNN (ResNet-50) auf zuvor bereinigten Car-Typ-Ordnern,
um das Modell später für Pseudo-Labeling auf den restlichen Daten zu verwenden.

Voraussetzungen:
 - Für jede Klasse ein Ordner mit Trainingsbildern

Konfiguration:
  CLASS_DIRS = {
    'SUV': Path('path/to/SUV_folder'),
    'Sedan': Path('path/to/Sedan_folder'),
    # ... weitere Klassen
  }
  VAL_SPLIT      = 0.2    # Anteil für Validation
  BATCH_SIZE     = 32
  NUM_EPOCHS     = 20
  LR             = 1e-4
  EARLY_STOPPING = 5      # Stoppt, wenn Val-Accuracy sich nicht verbessert
  MODEL_OUT      = Path('models/car_classifier.pth')
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image
from tqdm import tqdm
#from torch.optim.lr_scheduler import OneCycleLR
from torch.distributions import Beta

# === KONFIGURATION ===
CLASS_DIRS = {
    # Beispiel:
    'Coupe': Path(r'C:\Users\Erik\HLCVProject\new_data\highway_car\Coupe'),
    'Hatchback': Path(r'C:\Users\Erik\HLCVProject\new_data\highway_car\Hatchback'),
    'Minivan': Path(r'C:\Users\Erik\HLCVProject\new_data\highway_car\Minivan'),
    'Pickup': Path(r'C:\Users\Erik\HLCVProject\new_data\highway_car\Pickup'),
    'Van': Path(r'C:\Users\Erik\HLCVProject\new_data\highway_car\Van'),
    'SUV': Path(r'C:\Users\Erik\HLCVProject\new_data\highway_car\SUV'),
    'Sedan': Path(r'C:\Users\Erik\HLCVProject\new_data\highway_car\Sedan'),
    'Wagon': Path(r'C:\Users\Erik\HLCVProject\new_data\highway_car\Wagon')
    # Füge hier alle deine Klassen hinzu
}
VAL_SPLIT      = 0.2
BATCH_SIZE     = 32
NUM_EPOCHS     = 50
LR             = 1e-4
EARLY_STOPPING = 10
MODEL_OUT      = Path(r'C:\Users\Erik\HLCVProject\Project\models\car_classifier4.pth')
MIXUP_ALPHA = 0.2
# ======================

# 1) MixUp-Helper definieren
def mixup_data(x, y, alpha=0.2, device='cpu'):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Custom Dataset
class CarDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# Lade Samples
def load_samples(class_dirs):
    samples = []
    class_to_idx = {}
    for idx, (cls, folder) in enumerate(class_dirs.items()):
        class_to_idx[cls] = idx
        if not folder.exists():
            raise RuntimeError(f"Ordner nicht gefunden: {folder}")
        for img_path in folder.glob('*'):
            if img_path.suffix.lower() in ('.jpg','.jpeg','.png'):
                samples.append((str(img_path.resolve()), idx))
    return samples, class_to_idx

# Main
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 1) Samples und Label-Map laden
    samples, class_to_idx = load_samples(CLASS_DIRS)
    paths, labels = zip(*samples)
    # 2) Split
    train_idx, val_idx = train_test_split(
        list(range(len(samples))), test_size=VAL_SPLIT,
        stratify=labels, random_state=42
    )
    train_samples = [samples[i] for i in train_idx]
    val_samples   = [samples[i] for i in val_idx]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # 3) Transforms
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

    train_ds = CarDataset(train_samples, tfm_train)
    val_ds   = CarDataset(val_samples,   tfm_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,num_workers=4)

    # 4) Modell initialisieren
    PRETRAINED_PATH = Path(r'C:\Users\Erik\HLCVProject\Project\models\resnet50-0676ba61.pth')  # dein heruntergeladener Checkpoint
    model = models.resnet50(pretrained=False)
    # FC-Layer neu definieren
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_DIRS))
    # Gewichte laden
    state = torch.load(PRETRAINED_PATH, map_location='cpu')
    # Entferne alte fc-Gewichte, falls vorhanden
    state.pop('fc.weight', None)
    state.pop('fc.bias', None)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    #optimizer = optim.Adam(model.parameters(), lr=LR)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-2
    )
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    #max_lr = 1e-3
    #div_factor = max_lr / 1e-4  # =10.0

    #scheduler = OneCycleLR(
    #    optimizer,
    #    max_lr=max_lr,
    #    steps_per_epoch=len(train_loader),
    #    epochs=NUM_EPOCHS,
    #    pct_start=0.3,  # 30% des Zyklus für den Anstieg
    #    anneal_strategy='cos',  # Cosine‑Annealing
    #    div_factor=div_factor,  # initial_lr = max_lr/div_factor = 1e-4
    #    final_div_factor=1e4  # End‑LR = initial_lr/final_div_factor
    #)

    # 5) Training mit Early Stopping und LR-Decay
    best_acc = 0.0
    no_improve = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        # Training
        model.train()
        for imgs, labs in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]"):
            imgs, labs = imgs.to(device), labs.to(device)
            inputs, targets_a, targets_b, lam = mixup_data(imgs, labs, MIXUP_ALPHA, device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()
            # LR-Schritt
        #scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Aktuelle Lernrate: {current_lr:.2e}")
        #current_lr = scheduler.get_last_lr()[0]
        #print(f"Nach Epoch {epoch}: Learning Rate = {current_lr:.2e}")
        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labs in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]  "):
                imgs, labs = imgs.to(device), labs.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labs).sum().item()
        acc = correct / len(val_ds)
        print(f"Epoch {epoch}: Val Acc = {acc:.4f}")
        # Early Stopping
        if acc > best_acc:
            best_acc = acc
            no_improve = 0
            MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUT)
        else:
            no_improve += 1
            if no_improve >= EARLY_STOPPING:
                print(f"Keine Verbesserung in {EARLY_STOPPING} Epochen. Stoppe Training.")
                break
    print(f"Fertig. Best Val Acc = {best_acc:.4f}. Model gespeichert in {MODEL_OUT}.")


if __name__ == '__main__':
    main()