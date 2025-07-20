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
    'SUV','Sedan','Wagon','Coupe',
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
    for col in ('crop_name','path','xmin','ymin','xmax','ymax'):
        if col not in df.columns:
            print(f"Fehler: Spalte '{col}' fehlt in CSV", file=sys.stderr)
            sys.exit(1)

    # 2) Fehlende not_bus_truck-Einträge hinzufügen
    existing = set(df['crop_name'])
    new_records = []
    for img in NOTBUS_DIR.glob('*'):
        if img.suffix.lower() not in ('.jpg','.jpeg','.png'):
            continue
        if img.name not in existing:
            new_records.append({
                'day': None,
                'orig_image': None,
                'crop_name': img.name,
                'xmin': None, 'ymin': None, 'xmax': None, 'ymax': None,
                'width': None, 'height': None,
                'path': str(img.resolve()),
                'coarse_type': 'not_bus_truck'
            })
    if new_records:
        df = pd.concat([df, pd.DataFrame(new_records)], ignore_index=True)
        df.to_csv(HIGHWAY_CSV, index=False)
        print(f"CSV überschrieben: {len(new_records)} neue Einträge hinzugefügt.")
    else:
        print("Keine neuen 'not_bus_truck'-Einträge gefunden.")

    # 3) Filter nach Bounding-Box-Seitenverhältnis (≤5:1)
    # Berechne Box-Breite und -Höhe
    df['box_w'] = df['xmax'] - df['xmin']
    df['box_h'] = df['ymax'] - df['ymin']
    before = len(df)
    df = df[df['box_h'] > 0]  # vermeide Division durch Null
    df = df[df['box_w'] / df['box_h'] <= 5]
    after = len(df)
    print(f"Gefiltert nach Seitenverhältnis ≤5:1 – {before-after} Bilder entfernt, {after} verbleibend.")

    # 4) Sampling und Kopieren
    # Einmalige Liste aller Datensätze mischen
    records = df[['crop_name','path','coarse_type']].drop_duplicates().to_dict('records')
    random.shuffle(records)

    # Zielordner neu anlegen
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for vt in TYPES:
        (OUTPUT_DIR/vt).mkdir(parents=True, exist_ok=True)

    # Pro Typ unique Samples entnehmen
    used = set()
    for vt in TYPES:
        count = 0
        for rec in records:
            if count >= SAMPLE_PER_CLASS:
                break
            if rec['crop_name'] in used:
                continue
            # nur Einträge matching vt (für negative 'not_bus_truck' nutzen coarse_type)
            src = Path(rec['path'])
            if not src.exists():
                continue
            dst = OUTPUT_DIR/vt/rec['crop_name']
            shutil.copy(src, dst)
            used.add(rec['crop_name'])
            count += 1
        print(f"{vt}: {count} Bilder kopiert nach '{OUTPUT_DIR/vt}'")

    print("\nFertig: Manuelle Car-Samples erstellt.")

if __name__ == '__main__':
    main()