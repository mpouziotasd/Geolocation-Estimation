import torch
torch.classes.__path__ = []

from ultralytics import YOLO


def detect(model, frame, device='cpu'):    
    results = model(frame, imgsz=3840, device=device)
    return results

def get_det_coordinates(xyxy):
    x1, y1, x2, y2 = xyxy
    width = x2 - x1
    height = y2 - y1
    x_center = x1 + width // 2
    y_center = y1 + height // 2

    return width, height, x_center, y_center

