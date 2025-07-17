#!/usr/bin/env python3
"""
pseudo_label_highway.py

Erzeugt Pseudo-Labels für Highway-Crops (Cars, Trucks, Buses), teilt sie stratified
in Train (70%), Val (20%) und Test (10%) auf, sortiert alle Crops in
`data/highway/sorted/{train,val,test}/{vehicle_type}/` und legt eine Übersicht-CSV an.
Fortschritt wird via tqdm angezeigt.

Voraussetzungen:
 - `highway_car_crops.csv` mit Spalten ['crop_name','path']
 - `highway_truck_bus_crops.csv` mit Spalten ['crop_name','coarse_type','path']
 - `models/stanford_teacher.pth` (feingetuntes ResNet-50 auf Stanford Cars)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

# === KONFIGURATION ===
CAR_CSV         = Path(r"C:\Users\Erik\Desktop\Uni\HLCV\Project\data\highway\highway_car_crops.csv")
TRUCKBUS_CSV    = Path(r"C:\Users\Erik\Desktop\Uni\HLCV\Project\data\highway\highway_truck_bus_crops.csv")
SORTED_DIR      = Path(r"C:\Users\Erik\Desktop\Uni\HLCV\Project\data\highway\sorted")
TEACHER_PTH     = Path(r"C:\Users\Erik\Desktop\Uni\HLCV\Project\models\stanford_teacher_ep100v2.pth")
DEVICE        = "cuda:0" if torch.cuda.is_available() else "cpu"
CONF_THRESH   = 0.7
INPUT_SIZE    = 224
# Datenaufteilung
TEST_RATIO    = 0.10   # 10%
VAL_RATIO     = 0.20   # 20%
TRAIN_RATIO   = 0.70   # 70%
# ======================

def load_teacher(path: Path, num_classes: int, device: str) -> torch.nn.Module:
    model = models.resnet50(pretrained=False)
    state = torch.load(path, map_location="cpu")
    state.pop('fc.weight', None)
    state.pop('fc.bias', None)
    model.load_state_dict(state, strict=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device).eval()

# Bildtransforms
tfm = transforms.Compose([
    transforms.Resize(INPUT_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def stratified_split(df: pd.DataFrame, label_col: str):
    df_rest, df_test = train_test_split(
        df, test_size=TEST_RATIO,
        stratify=df[label_col], random_state=42
    )
    rel_val = VAL_RATIO / (1 - TEST_RATIO)
    df_train, df_val = train_test_split(
        df_rest, test_size=rel_val,
        stratify=df_rest[label_col], random_state=42
    )
    return df_train, df_val, df_test

if __name__ == "__main__":
    records = []

    # --- 1) Trucks & Buses ---
    #df_tb = pd.read_csv(TRUCKBUS_CSV).rename(columns={'coarse_type':'vehicle_type'})[['path','vehicle_type']]
    #tb_train, tb_val, tb_test = stratified_split(df_tb, 'vehicle_type')
    #for split_name, df_split in [('train', tb_train), ('val', tb_val), ('test', tb_test)]:
    #    for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"[Trucks/Buses] {split_name}", unit="crop"):
    #        src = Path(row['path'])
    #        cls = row['vehicle_type'].capitalize()
    #        dst = SORTED_DIR / split_name / cls / src.name
    #        dst.parent.mkdir(parents=True, exist_ok=True)
    #        shutil.copy(src, dst)
    #        records.append({'crop_name': src.name, 'dest_path': str(dst), 'vehicle_type': cls, 'split': split_name})

    # --- 2) Cars: Pseudo-Label mit ResNet-Teacher ---
    stanford_meta = pd.read_csv(r"C:\Users\Erik\Desktop\Uni\HLCV\Project\data\stanford\annotations_stanford_all.csv")
    car_types = sorted(stanford_meta['vehicle_type'].unique())
    teacher = load_teacher(TEACHER_PTH, len(car_types), DEVICE)

    df_cars = pd.read_csv(CAR_CSV)
    pseudo = []
    # 1) Vor der Loop eine Liste anlegen
    conf_list = []

    # 2) In der Inferenz-Loop nur die Python-Floats sammeln
    for _, row in tqdm(df_cars.iterrows(), total=len(df_cars), desc="[Cars → Inference]", unit="crop"):
        src = Path(row['path'])
        if not src.exists():
            continue
        img = Image.open(src).convert('RGB')
        x = tfm(img).unsqueeze(0).to(DEVICE)
        probs = F.softmax(teacher(x), dim=1)
        conf, idx = probs.max(1)
        conf_val = conf.item()  # .item() gibt Float auf CPU
        conf_list.append(conf_val)
        if conf_val >= CONF_THRESH:
            pseudo.append({'path': str(src), 'vehicle_type': car_types[idx.item()]})

    # 3) Nach der Loop eine Pandas-Series draus machen und beschreiben
    import pandas as pd

    s = pd.Series(conf_list)
    print("Confidences (min, 25%, 50%, 75%, max):")
    print(s.quantile([0, .25, .5, .75, 1.0]))
    print("Mean:", s.mean(), "Std:", s.std())

    # 4) Jetzt kannst du df_pseudo bauen wie gehabt
    df_pseudo = pd.DataFrame(pseudo)
    print(f"Car-Pseudo-Labels erzeugt: {len(df_pseudo)}")

    # --- 3) Cars splitten und kopieren ---
    car_train, car_val, car_test = stratified_split(df_pseudo, 'vehicle_type')
    for split_name, df_split in [('train', car_train), ('val', car_val), ('test', car_test)]:
        for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"[Cars] {split_name}", unit="crop"):
            src = Path(row['path'])
            cls = row['vehicle_type']
            dst = SORTED_DIR / split_name / cls / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
            records.append({'crop_name': src.name, 'dest_path': str(dst), 'vehicle_type': cls, 'split': split_name})

    # --- 4) Übersicht-CSV speichern ---
    df_overview = pd.DataFrame(records)
    csv_out = SORTED_DIR / 'highway_assignment_overview.csv'
    df_overview.to_csv(csv_out, index=False)
    print(f"✅ Übersicht-CSV gespeichert in '{csv_out}'")