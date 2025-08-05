from ultralytics import YOLO
import cv2
import numpy as np  # Needed for mask conversion

# Load your trained model (use best.pt or last.pt)
model_path = './runs/segment/train/weights/best.pt'  # or 'last.pt'
model = YOLO(best.pt)

# Load your test image
image_path = './data/images/val/748.jpg'
img = cv2.imread(image_path)
H, W, _ = img.shape

# Run inference
results = model(img)

# Process and save masks
for result in results:
    if result.masks is not None:
        for j, mask in enumerate(result.masks.data):
            mask_np = mask.cpu().numpy() * 255  # convert tensor to numpy
            mask_resized = cv2.resize(mask_np, (W, H))
            mask_resized = mask_resized.astype(np.uint8)
            cv2.imwrite(f'./output_mask_{j}.png', mask_resized)
