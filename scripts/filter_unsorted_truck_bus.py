#!/usr/bin/env python3
"""
filter_unsorted_truck_bus.py

Verwendet das feingetunte Truck/Bus/Other-ResNet-50 Modell, um alle Bilder
aus mehreren unsortierten Ordnern zu klassifizieren und in separate
Klassenordner zu verschieben:
  - truck
  - bus
  - not_truck_bus
unter `OUTPUT_DIR/{truck,bus,not_truck_bus}/`.

Konfiguration:
  UNSORTED_DIRS = [Path('data/highway/unsorted1'), Path('data/highway/unsorted2'), ...]
  OUTPUT_DIR   = Path('data/highway/filtered')  # Ausgabe-Ordner für gefilterte Klassen
  MODEL_PTH    = Path('models/truck_bus_resnet50.pth')
  BATCH_SIZE   = 1                             # Einzel-Inferenz pro Bild
  DEVICE       = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  THRESH       = 0.5                           # Confidence-Schwelle
"""
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import shutil
from tqdm import tqdm
import pandas as pd

# === KONFIGURATION ===
UNSORTED_DIRS = [
    Path(r'C:\Users\Erik\Desktop\Uni\HLCV\Project\data\highway\sorted\val\bus'),
    Path(r'C:\Users\Erik\Desktop\Uni\HLCV\Project\data\highway\sorted\val\truck'),
    Path(r'C:\Users\Erik\Desktop\Uni\HLCV\Project\data\highway\sorted\train\bus'),
    Path(r'C:\Users\Erik\Desktop\Uni\HLCV\Project\data\highway\sorted\train\truck')
    # weitere Ordner hier hinzufügen, z.B.:
    # Path('data/highway/unsorted_extra1'),
    # Path('data/highway/unsorted_extra2'),
]
OUTPUT_DIR   = Path(r'C:\Users\Erik\Desktop\Uni\HLCV\Project\data\highway\filtered')
MODEL_PTH    = Path(r'C:\Users\Erik\Desktop\Uni\HLCV\Project\models\truck_bus_resnet50.pth')
DEVICE       = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESH       = 0.95
# ======================

# Bild-Transform für Inferenz
tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Klassen-Reihenfolge muss mit Training übereinstimmen
CLASSES = ['truck', 'bus', 'not_truck_bus']

# Modell laden (Struktur wie beim Training)
def load_model(path: Path, device: str):
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state)
    return model.to(device).eval()

if __name__ == '__main__':
    # Ausgabe-Ordner erstellen
    for cls in CLASSES:
        (OUTPUT_DIR/cls).mkdir(parents=True, exist_ok=True)

    # Modell initialisieren
    model = load_model(MODEL_PTH, DEVICE)

    # Alle Bilder in den unsortierten Verzeichnissen sammeln
    img_paths = []
    for unsorted_dir in UNSORTED_DIRS:
        img_paths += [p for p in unsorted_dir.rglob('*')
                      if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]

    conf_list = []
    # Inferenz und Klassifizierung
    for img_path in tqdm(img_paths, desc='Classifying'):  # Fortschrittsbalken
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            continue
        x = tfm(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            conf, idx = probs.max(1)
            conf_val = conf.item()
            conf_list.append(conf_val)
            cls = CLASSES[idx.item()]
            # Niedrige Confidence behandeln
            if conf_val < THRESH:
                cls = 'not_truck_bus'
        # Bild kopieren
        dst = OUTPUT_DIR / cls / img_path.name
        shutil.copy(img_path, dst)

    s = pd.Series(conf_list)
    print("\nConfidences (min, 25%, 50%, 75%, max):")
    print(s.quantile([0, .25, .5, .75, 1.0]))
    print(f"Mean: {s.mean():.4f}, Std: {s.std():.4f}")
    print(f"\nBilder klassifiziert und kopiert nach '{OUTPUT_DIR}'.")
