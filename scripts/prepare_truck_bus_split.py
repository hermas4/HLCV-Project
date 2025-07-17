#!/usr/bin/env python3
"""
prepare_truck_bus_split.py

Liest bereinigte Ordner für die Klassen 'truck' und 'bus' ein,
führt für jede Klasse einen stratified Split (70% train, 20% val, 10% test)
durch und kopiert die Bilder in
`data/highway/sorted/{train,val,test}/{vehicle_type}/`.
Schreibt zudem eine CSV mit den Spalten:
  ['crop_name','path','vehicle_type']

Konfiguration direkt im Skript anpassen.
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
import sys

# === KONFIGURATION ===
# Listen mit den bereinigten Ordnerpfaden für Truck und Bus
TRUCK_DIRS = [
    Path(r'C:\Users\Erik\Desktop\Uni\HLCV\Project\data\highway\filtered\truck'),
]
BUS_DIRS   = [
    Path(r'C:\Users\Erik\Desktop\Uni\HLCV\Project\data\highway\filtered\bus'),
]
# Ausgabe-CSV (optional)
CSV_OUT     = Path(r'C:\Users\Erik\Desktop\Uni\HLCV\Project\data\highway\highway_truck_bus_crops.csv')
# Basis-Ordner für sortierte Bilder
TARGET_BASE = Path(r'C:\Users\Erik\Desktop\Uni\HLCV\Project\data\highway\sorted')
# Split-Ratios 70/20/10
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10
# Zufalls-Seed für Repro
RANDOM     = 42
# ======================

def collect_images(dirs, vtype):
    """Sammelt alle Bildpfade und markiert mit vehicle_type und crop_name."""
    recs = []
    for folder in dirs:
        if not folder.exists():
            print(f"Warnung: Ordner nicht gefunden: {folder}", file=sys.stderr)
            continue
        for img in folder.glob('*'):
            if img.is_file() and img.suffix.lower() in ('.jpg','.jpeg','.png'):
                recs.append({
                    'crop_name': img.name,
                    'path':      str(img.resolve()),
                    'vehicle_type': vtype
                })
    return recs


def stratified_split(df):
    """Stratified 70/20/10 Split für DataFrame mit 'vehicle_type'."""
    # Test abspalten
    df_trainval, df_test = train_test_split(
        df, test_size=TEST_RATIO, stratify=df['vehicle_type'], random_state=RANDOM
    )
    # Val aus trainval
    rel_val = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    df_train, df_val = train_test_split(
        df_trainval, test_size=rel_val, stratify=df_trainval['vehicle_type'], random_state=RANDOM
    )
    return df_train, df_val, df_test


def copy_split(df_split, split_name):
    """Kopiert Bilder in TARGET_BASE/{split_name}/{vehicle_type}/."""
    for _, row in df_split.iterrows():
        src = Path(row['path'])
        dst = TARGET_BASE / split_name / row['vehicle_type'] / row['crop_name']
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)


def main():
    # 1) Records sammeln für Truck und Bus
    records = []
    records += collect_images(TRUCK_DIRS, 'truck')
    records += collect_images(BUS_DIRS,   'bus')
    df = pd.DataFrame(records)
    print(f"Gefundene Bilder insgesamt: {len(df)}")

    # 2) CSV schreiben
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False, columns=['crop_name','path','vehicle_type'])
    print(f"CSV geschrieben: {CSV_OUT}")

    # 3) Split durchführen
    df_train, df_val, df_test = stratified_split(df)
    print(f"Anzahl Train/Val/Test: {len(df_train)}/{len(df_val)}/{len(df_test)}")

    # 4) Alte Struktur löschen und Ordner anlegen
    if TARGET_BASE.exists():
        shutil.rmtree(TARGET_BASE)
    for split in ('train','val','test'):
        for cls in ('truck','bus'):
            (TARGET_BASE / split / cls).mkdir(parents=True, exist_ok=True)

    # 5) Kopieren
    copy_split(df_train, 'train')
    copy_split(df_val,   'val')
    copy_split(df_test,  'test')
    print(f"Bilder sortiert nach {TARGET_BASE}/ (train/val/test).")

if __name__ == '__main__':
    main()
