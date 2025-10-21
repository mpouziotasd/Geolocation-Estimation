import cv2 as cv


"""
Visualization Brainstorm:
    - Remove middle areas
    - 
"""
def draw_data(frame, bboxes, labels, cls_id, custom_labels=None, color=(0, 255, 0)):
    for i, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = map(int, bbox)
        box_w = x_max - x_min
        box_h = y_max - y_min
        percent = 0.15
        corner_len_w = int(box_w * percent)
        corner_len_h = int(box_h * percent)

        # Top-left
        cv.line(frame, (x_min, y_min), (x_min + corner_len_w, y_min), color, 3)
        cv.line(frame, (x_min, y_min), (x_min, y_min + corner_len_h), color, 3)
        # Top-right
        cv.line(frame, (x_max, y_min), (x_max - corner_len_w, y_min), color, 3)
        cv.line(frame, (x_max, y_min), (x_max, y_min + corner_len_h), color, 3)
        # Bottom-left
        cv.line(frame, (x_min, y_max), (x_min + corner_len_w, y_max), color, 3)
        cv.line(frame, (x_min, y_max), (x_min, y_max - corner_len_h), color, 3)
        # Bottom-right
        cv.line(frame, (x_max, y_max), (x_max - corner_len_w, y_max), color, 3)
        cv.line(frame, (x_max, y_max), (x_max, y_max - corner_len_h), color, 3)

        # --- Draw text ---
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2

        if custom_labels is not None:
            text = custom_labels[i]  # use your label with ID
        else:
            text = cls_id[int(labels[i])]  # fallback to class name

        text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
        text_x = x_min
        text_y = y_min - 10 if y_min - 10 > 10 else y_min + text_size[1] + 10
        cv.putText(frame, text, (text_x + 2, text_y), font, font_scale, (255, 255, 255), font_thickness, cv.LINE_AA)

    return frame
