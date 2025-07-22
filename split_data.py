import os
import random
import shutil

source_img_dir = "cards_dataset/images"
source_label_dir = "cards_dataset/labels"

destination = "cards_yolo_split"
os.makedirs(f"{destination}/train/images", exist_ok=True)
os.makedirs(f"{destination}/train/labels", exist_ok=True)
os.makedirs(f"{destination}/valid/images", exist_ok=True)
os.makedirs(f"{destination}/valid/labels", exist_ok=True)

images = [f for f in os.listdir(source_img_dir) if f.endswith(".jpg")]
random.shuffle(images)

split_idx = int(0.8 * len(images))
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

for img_list, split in [(train_imgs, "train"), (val_imgs, "valid")]:
	for img in img_list:
		lbl = img.replace(".jpg", ".txt")
		shutil.copy(f"{source_img_dir}/{img}", f"{destination}/{split}/images/{img}")
		shutil.copy(f"{source_label_dir}/{lbl}", f"{destination}/{split}/labels/{lbl}")
