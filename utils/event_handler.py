import cv2 as cv

event_state = {"last_pos": None}

def on_click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        event_state['last_pos'] = (x, y)

def get_last_click():
    pos = event_state["last_pos"]
    event_state["last_pos"] = None
    return pos