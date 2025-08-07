import pandas as pd
import os
import shutil
from tqdm import tqdm

excel_path = "ripeness_data.xlsx"
image_dir = "all_images"
output_dir = "dataset"

static_bbox = [
    [0, 0.49276315789473685, 0.47434210526315795, 0.475, 0.7723684210526316]
]

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

splits = ['train', 'valid', 'test']
for split in splits:
    os.makedirs(f"{output_dir}/{split}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/{split}/labels", exist_ok=True)

df = pd.read_excel(excel_path)

df["class_id"] = df["Ripening Index Classification"] - 1

df = df.sample(frac=1).reset_index(drop=True)

total = len(df)
train_end = int(total * train_ratio)
val_end = train_end + int(total * val_ratio)

train_df = df[:train_end]
val_df = df[train_end:val_end]
test_df = df[val_end:]

def process_split(split_name, split_df):
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Processing {split_name}"):
        filename = row["File Name"]
        class_id = row["class_id"]
        image_path = os.path.join(image_dir, filename + ".jpg")
        if not os.path.exists(image_path):
            print("Missing:", image_path)
            continue

        dest_img_path = os.path.join(output_dir, split_name, "images", filename + ".jpg")
        shutil.copy(image_path, dest_img_path)

        label_path = os.path.join(output_dir, split_name, "labels", filename + ".txt")
        with open(label_path, "w") as f:
            for bbox in static_bbox:
                bbox_str = " ".join(str(x) for x in bbox)
                f.write(f"{class_id} {bbox_str}\n")

process_split("train", train_df)
process_split("valid", val_df)
process_split("test", test_df)

yaml_path = os.path.join(output_dir, "data.yaml")
with open(yaml_path, "w") as f:
    f.write("train: ../train/images\n")
    f.write("val: ../valid/images\n")
    f.write("test: ../test/images\n\n")
    f.write("nc: 4\n")
    f.write("names: ['0', '1', '2', '3']\n")

print("✅ DONE, bro. Dataset is YOLOv8-ready.")
