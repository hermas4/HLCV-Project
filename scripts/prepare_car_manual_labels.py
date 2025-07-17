#!/usr/bin/env python3
"""
prepare_car_manual_labels.py

Aktualisiert zuerst die 'highway_car_crops.csv' um fehlende Einträge
für 'not_bus_truck' aus einem Ordner und überschreibt die CSV.
Anschließend sampelt es bis zu SAMPLE_PER_CLASS zufällige Bilder
pro Fahrzeugtyp (aus TYPES) und kopiert sie in OUTPUT_DIR/{type}/.

Konfiguration:
  HIGHWAY_CSV      = Path('data/highway/highway_car_crops.csv')
  NOTBUS_DIR       = Path('data/highway/cleaned/negative')
  OUTPUT_DIR       = Path('data/highway/manual_car_samples')
  SAMPLE_PER_CLASS = 1000
  TYPES            = [
    'SUV','Sedan','Wagon','Coupe','Convertible',
    'Hatchback','Truck','Van','Minivan','Pickup','not_bus_truck'
  ]
"""
import random
import pandas as pd
from pathlib import Path
import shutil
import sys

# === KONFIGURATION ===
HIGHWAY_CSV      = Path(r'C:\Users\Erik\Desktop\Uni\HLCV\Project\highway_car_crops.csv')
NOTBUS_DIR       = Path(r'C:\Users\Erik\Desktop\Uni\HLCV\new_data\highway\filtered\ac_not_truck_bus')
OUTPUT_DIR       = Path(r'C:\Users\Erik\Desktop\Uni\HLCV\new_data\highway_car')
SAMPLE_PER_CLASS = 1000
TYPES = [
    'SUV','Sedan','Wagon','Coupe','Convertible',
    'Hatchback','Van','Minivan','Pickup'
]
# ======================

def main():
    random.seed(42)

    # 1) CSV einlesen
    if not HIGHWAY_CSV.exists():
        print(f"Fehler: CSV nicht gefunden: {HIGHWAY_CSV}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(HIGHWAY_CSV)
    # Prüfen notwendiger Spalten
    for col in ('crop_name','path'):
        if col not in df.columns:
            print(f"Fehler: Spalte '{col}' fehlt in CSV", file=sys.stderr)
            sys.exit(1)

    # 2) Prepare list of all image paths (unique)
    all_records = df[['crop_name','path']].drop_duplicates().reset_index(drop=True)
    available = all_records.to_dict('records')
    random.shuffle(available)

    # 3) For each type, sample unique images
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for vtype in TYPES:
        dest = OUTPUT_DIR / vtype
        dest.mkdir(parents=True, exist_ok=True)
        samples = []
        for _ in range(min(SAMPLE_PER_CLASS, len(available))):
            samples.append(available.pop(0))
        for rec in samples:
            src = Path(rec['path'])
            dst = dest / rec['crop_name']
            if not src.exists():
                print(f"Warnung: Datei existiert nicht: {src}", file=sys.stderr)
                continue
            shutil.copy(src, dst)
        print(f"{vtype}: {len(samples)} Bilder kopiert nach '{dest}'.")

    print("Fertig: Manuelle Car-Samples erstellt.")

if __name__ == '__main__':
    main()