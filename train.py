from ultralytics import YOLO

model = YOLO(best.pt)  # load a pretrained model (recommended for training)

model.train(data='config.yaml', epochs=10, imgsz=640)
