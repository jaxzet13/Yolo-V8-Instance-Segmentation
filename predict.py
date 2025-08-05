from ultralytics import YOLO
import cv2
import numpy as np  # Required for handling numpy arrays

model_path = './runs/segment/train/weights/last.pt'
image_path = './data/images/val/748.jpg'

img = cv2.imread(image_path)
H, W, _ = img.shape

model = YOLO(best.pt)

results = model(img)

for result in results:
    for j, mask in enumerate(result.masks.data):

        mask = mask.numpy() * 255

        mask = cv2.resize(mask, (W, H))

        cv2.imwrite('./output.png', mask)
