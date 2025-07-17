#!/usr/bin/env python3
"""
prepare_stanford.py

Script zum stratified Split der vorhandenen Stanford-Crops in Train/Val/(Test)-Ordner.
Voraussetzungen:
 - annotations_stanford_all.csv mit Spalten ['image_id','vehicle_type','path']
 - 'path'-Spalte enthält den absoluten Pfad zu jedem Crop-Bild

Konfiguration erfolgt direkt in dieser Datei.
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
import sys

# === Konfiguration ===
# Pfad zur Annotation-CSV (muss Spalte 'path' enthalten)
STANFORD_CSV = Path(r"C:\Users\Erik\Desktop\Uni\HLCV\Project\data\stanford\annotations_stanford_all.csv")
# Ziel-Verzeichnis für die train/val/(test)-Ordner
OUT_DIR      = Path(r"C:\Users\Erik\Desktop\Uni\HLCV\Project\data\stanford\crops")
# Anteil für Validation
VAL_SPLIT    = 0.2  # z.B. 0.2 für 20%
# Anteil für Test (0 = kein Test-Split)
TEST_SPLIT   = 0.0
# ======================

def make_dirs_for_types(base: Path, splits, types):
    """
    Legt für jeden Split und jeden Fahrzeugtyp einen Ordner an.
    """
    for split in splits:
        for t in types:
            (base / split / t).mkdir(parents=True, exist_ok=True)


def copy_crops(df: pd.DataFrame, out_dir: Path, split: str):
    """
    Kopiert jede Datei anhand des absoluten Pfades in df['path']
    in das Zielverzeichnis out_dir/{split}/{vehicle_type}/
    """
    for _, row in df.iterrows():
        src = Path(row['path'])
        if not src.exists():
            print(f"Warnung: Datei nicht gefunden: {src}", file=sys.stderr)
            continue
        dst = out_dir / split / row['vehicle_type'] / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)


def main():
    # CSV einlesen
    df = pd.read_csv(STANFORD_CSV)
    # Spalten validieren
    required = {'image_id', 'vehicle_type', 'path'}
    if not required.issubset(df.columns):
        print(f"CSV muss die Spalten {required} enthalten.", file=sys.stderr)
        sys.exit(1)

    # Fahrzeugtypen, in gewünschter Reihenfolge
    types = [
        'SUV', 'Sedan', 'Wagon', 'Coupe', 'Convertible',
        'Hatchback', 'Truck', 'Van', 'Minivan', 'Pick Up'
    ]
    splits = ['train', 'val'] + (['test'] if TEST_SPLIT > 0 else [])

    # Ordnerstruktur erstellen
    make_dirs_for_types(OUT_DIR, splits, types)

    # Test-Split
    if TEST_SPLIT > 0:
        train_val, test_df = train_test_split(
            df, test_size=TEST_SPLIT,
            stratify=df['vehicle_type'], random_state=42
        )
        df = train_val
        print(f"Test-Split: {len(test_df)} Crops")
        copy_crops(test_df, OUT_DIR, 'test')

    # Train/Val-Split
    train_df, val_df = train_test_split(
        df, test_size=VAL_SPLIT,
        stratify=df['vehicle_type'], random_state=42
    )
    print(f"Train: {len(train_df)} Crops, Val: {len(val_df)} Crops")

    # Dateien kopieren
    copy_crops(train_df, OUT_DIR, 'train')
    copy_crops(val_df,   OUT_DIR, 'val')

    print("✅ Stanford-Crops erfolgreich in Train/Val/(Test) unterteilt.")

if __name__ == '__main__':
    main()