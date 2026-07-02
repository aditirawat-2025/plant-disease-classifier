"""
Splits selected class folders from the PlantVillage 'color' directory
into data/train/<class> and data/val/<class> (80/20 split).

Edit SOURCE_DIR and SELECTED_CLASSES below to match your setup, then run:
    python split_dataset.py
"""

import os
import random
import shutil

# ---------- EDIT THESE ----------
# Path to the "color" folder inside your extracted PlantVillage dataset
SOURCE_DIR = r"C:\Users\Jaswant\Desktop\plantvillage dataset\color"

# Which class folders to use — edit this list to add/remove classes
SELECTED_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
]

# Where your project's data folder is (relative to this script)
DEST_DIR = "data"
TRAIN_SPLIT = 0.8  # 80% train, 20% val
# ---------------------------------

random.seed(42)  # for reproducible splits

for class_name in SELECTED_CLASSES:
    src_class_dir = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(src_class_dir):
        print(f"⚠️  Skipping '{class_name}' — folder not found at {src_class_dir}")
        continue

    images = [f for f in os.listdir(src_class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_SPLIT)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    train_dest = os.path.join(DEST_DIR, "train", class_name)
    val_dest = os.path.join(DEST_DIR, "val", class_name)
    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(val_dest, exist_ok=True)

    for img in train_images:
        shutil.copy2(os.path.join(src_class_dir, img), os.path.join(train_dest, img))
    for img in val_images:
        shutil.copy2(os.path.join(src_class_dir, img), os.path.join(val_dest, img))

    print(f"✅ {class_name}: {len(train_images)} train, {len(val_images)} val")

print("\nDone! Your data/train and data/val folders are ready.")
