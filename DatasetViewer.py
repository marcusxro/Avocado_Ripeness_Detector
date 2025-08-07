import os
import cv2
import matplotlib.pyplot as plt

image_folder = "dataset/train/images"
label_folder = "dataset/train/labels"

image_exts = ['.jpg', '.jpeg', '.png']
images = [
    f for f in os.listdir(image_folder)
    if os.path.splitext(f)[1].lower() in image_exts
][:10] 

for img_file in images:
    img_path = os.path.join(image_folder, img_file)
    label_path = os.path.join(label_folder, os.path.splitext(img_file)[0] + '.txt')

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x_center, y_center, bbox_w, bbox_h = map(float, parts)

        x1 = int((x_center - bbox_w / 2) * w)
        y1 = int((y_center - bbox_h / 2) * h)
        x2 = int((x_center + bbox_w / 2) * w)
        y2 = int((y_center + bbox_h / 2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(img_file)
    plt.axis('off')
    plt.show()
