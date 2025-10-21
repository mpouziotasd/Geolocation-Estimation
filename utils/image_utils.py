import cv2 as cv

def overlay_gis(frame, overlay, shape):
    overlay = cv.cvtColor(overlay, cv.COLOR_RGB2BGR)


    H, W = frame.shape[:2]
    w, h = shape
    x1, y1 = 0, H - h
    x2, y2 = w, H

    frame[y1:y2, x1:x2] = overlay
    return frame
