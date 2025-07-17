from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import shutil
import os
import cv2
import pandas as pd

# 0. Parameter
OUTPUT_BASE   = Path(r"C:\Users\Erik\Desktop\Uni\HLCV\data\detected")
BASE_RAW_DIRS = [
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-09-27-14-43-04_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-09-30-14-41-23_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-09-30-15-03-39_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-09-30-15-19-35_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-10-04-13-52-40_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-10-04-14-22-41_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-10-10-15-17-24_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-10-10-15-24-37_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-10-10-15-32-33_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-10-10-15-35-18_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-10-10-16-00-11_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-10-10-16-12-20_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-10-10-16-43-45_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-10-10-18-25-04_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-10-10-18-41-33_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-10-26-12-49-56_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-10-26-13-00-25_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-10-26-13-04-33_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-10-26-17-55-06_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-10-26-17-57-22_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-10-26-18-03-11_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-10-26-18-22-27_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-10-26-18-38-03_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-10-30-09-53-48_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-10-30-10-01-47_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-10-30-10-04-51_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-10-30-10-24-32_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-10-30-10-26-40_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-11-01-10-07-39_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-11-01-10-20-23_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-11-02-18-05-08_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-11-03-15-03-15_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-11-03-15-28-03_bag",
    r"C:\Users\Erik\Desktop\Uni\HLCV\data\Autobahn_data\bluefox_2016-11-03-15-40-30_bag"
    # ...
    # deine Liste von Tagesordnern...
]
MODEL_PATH    = r"C:\Users\Erik\Desktop\Uni\HLCV\data\yolov8n.pt"
MIN_VEHICLES  = 3
MIN_WIDTH     = 80   # Mindestbreite
MIN_HEIGHT    = 80   # Mindesthöhe
# 1. Fahrzeug-Klassen
def is_vehicle(cls_name: str):
    VEHICLE_CLASSES = {'car', 'truck', 'bus'}
    return cls_name.lower() in VEHICLE_CLASSES

# 2. Output-Verzeichnis säubern
if OUTPUT_BASE.exists():
    shutil.rmtree(OUTPUT_BASE)
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# 3. YOLO laden
model = YOLO(MODEL_PATH)
print(f"Loaded YOLOv8 model with {model.model.yaml['nc']} classes")

# 4. Cropping & Metadaten sammeln
all_records = []
skipped_small = 0
counter = 0
for raw_dir in tqdm(BASE_RAW_DIRS, desc="Ordner"):
    day_name = Path(raw_dir).name
    out_dir  = OUTPUT_BASE / day_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(list(Path(raw_dir).glob("*.png")), desc=day_name, leave=False):
        # 4.1 Inferenz
        res = model.predict(source=str(img_path), save=False, verbose=False)[0]
        vehicle_boxes = [
            box.xyxy[0] for box, cls in zip(res.boxes, res.boxes.cls)
            if is_vehicle(res.names[int(cls)])
        ]
        # 4.2 Mindestanzahl prüfen
        if len(vehicle_boxes) < MIN_VEHICLES:
            continue

        # 4.3 Crops speichern & Metadaten
        for i, box in enumerate(vehicle_boxes):
            x1, y1, x2, y2 = map(int, box.tolist())
            crop = res.orig_img[y1:y2, x1:x2]
            h, w = crop.shape[:2]

            if w < MIN_WIDTH or h < MIN_HEIGHT or h > w:
                skipped_small += 1
                continue

            crop_name = f"{img_path.stem}_crop{i:02d}.jpg"
            crop_path = out_dir / crop_name
            cv2.imwrite(str(crop_path), crop)

            # Bestimme coarse_type über die Klasse von res.names[int(cls)]
            cls_name = res.names[int(res.boxes.cls[i])]
            coarse_type = cls_name.lower()  # "car", "truck" oder "bus"

            all_records.append({
                "day": day_name,
                "orig_image": img_path.name,
                "crop_name": crop_name,
                "xmin": x1,
                "ymin": y1,
                "xmax": x2,
                "ymax": y2,
                "width": w,
                "height": h,
                "path": str(crop_path),
                "coarse_type": coarse_type
            })

# 5a. DataFrame erzeugen
df_all = pd.DataFrame(all_records)

# 5b. Split nach coarse_type
df_cars       = df_all[df_all["coarse_type"] == "car"].reset_index(drop=True)
df_truck_bus  = df_all[df_all["coarse_type"].isin(["truck", "bus"])].reset_index(drop=True)

# 5c. CSVs schreiben
df_cars.to_csv("highway_car_crops.csv",      index=False)
df_truck_bus.to_csv("highway_truck_bus_crops.csv", index=False)

print(f"→ car-Crops:      {len(df_cars)} Einträge in highway_car_crops.csv")
print(f"→ truck/bus-Crops:{len(df_truck_bus)} Einträge in highway_truck_bus_crops.csv")

# 6. Abschluss-Statistik
print("\n=== Summary ===")
print(f"Total crops saved : {len(df_all)}")
print(f"Skipped small     : {skipped_small} (<{MIN_WIDTH}×{MIN_HEIGHT})")
print(f"Output CSV        : highway_all_days_crops_with_boxes.csv")