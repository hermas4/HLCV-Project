#!/usr/bin/env python3
"""
prepare_manual_labels.py

Erzeugt eine CSV-Datei `highway_manual_labels.csv` mit allen Highway-Car-Crops,
die nicht ausreichend sicher pseudo-gelabelt wurden (Confidence < CONF_THRESH),
um sie manuell mit `vehicle_type` zu versehen.

Voraussetzungen:
 - `highway_car_crops.csv` mit Spalten ['crop_name','path']
 - `highway_assignment_overview.csv` aus pseudo_label_highway.py (optional)

Nutzung:
    python prepare_manual_labels.py
"""
import pandas as pd
from pathlib import Path

# === KONFIGURATION ===
CAR_CSV      = Path(r"C:\Users\Erik\Desktop\Uni\HLCV\Project\data\highway\highway_car_crops.csv")
OVERVIEW_CSV = Path()
OUTPUT_CSV   = Path(r"C:\Users\Erik\Desktop\Uni\HLCV\Project\data\highway\highway_manual_labels.csv")
CONF_THRESH  = 0.9
# ======================

def main():
    # 1) Lade alle Car-Crops
    df_cars = pd.read_csv(CAR_CSV)

    # 2) Lade Übersicht falls vorhanden und filtere bereits verarbeitete
    if OVERVIEW_CSV.exists():
        df_assigned = pd.read_csv(OVERVIEW_CSV)
        # Nur Cars aus Overview
        df_assigned = df_assigned[df_assigned['dest_path'].str.contains("highway/sorted/")]
        # Entferne alle, die schon eine Zuweisung haben
        processed = set(Path(p).name for p in df_assigned['dest_path'])
        df_cars = df_cars[~df_cars['path'].apply(lambda p: Path(p).name in processed)]

    # 3) Erzeuge manuelle Label-Vorlage
    df_manual = df_cars.copy()
    df_manual['vehicle_type'] = ''  # leeres Feld zum Befüllen

    # 4) Speichere CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_manual.to_csv(OUTPUT_CSV, index=False, columns=['crop_name','path','vehicle_type'])
    print(f"✅ Manual-Label-Vorlage gespeichert in {OUTPUT_CSV} (Records: {len(df_manual)})")

if __name__ == '__main__':
    main()
