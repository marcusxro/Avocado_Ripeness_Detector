
import os
import shutil
import pandas as pd
import cv2
from tqdm import tqdm
from ultralytics import YOLO

excel_path = "ripeness_data.xlsx"
image_dir = "all_images"
output_dir = "dataset"
model_name = "yolov8n.pt"  

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


model = YOLO(model_name)


def process_split(split_name, split_df):
    """Process a dataset split (train/val/test) to create YOLO-formatted data
    """
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Processing {split_name}"):
        filename = row["File Name"]
        class_id = row["class_id"]

        img_path = os.path.join(image_dir, filename + ".jpg")
        if not os.path.exists(img_path):
            print(f"Missing file: {img_path}")
            continue

        dest_img_path = os.path.join(output_dir, split_name, "images", filename + ".jpg")
        shutil.copy(img_path, dest_img_path)

        results = model(img_path)

        detections = results[0].boxes


        label_path = os.path.join(output_dir, split_name, "labels", filename + ".txt")
        with open(label_path, "w") as f:
            for box in detections:

                xywh = box.xywhn.cpu().numpy().tolist()[0]  
                bbox_str = " ".join(f"{x:.6f}" for x in xywh)
                f.write(f"{class_id} {bbox_str}\n")

process_split("train", train_df)
process_split("valid", val_df)
process_split("test", test_df)


yaml_path = os.path.join(output_dir, "data.yaml")
with open(yaml_path, "w") as f:
    f.write("# YOLO dataset configuration\n")
    f.write(f"train: {output_dir}/train/images\n")
    f.write(f"val: {output_dir}/valid/images\n")
    f.write(f"test: {output_dir}/test/images\n\n")
    f.write(f"nc: {df['class_id'].nunique()}\n") 
    f.write("names: ['0', '1', '2', '3']\n")  

print("completed.")