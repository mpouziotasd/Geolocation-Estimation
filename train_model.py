from ultralytics import YOLO

model = YOLO("yolo11x.pt")

results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640)