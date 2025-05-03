import cv2
import numpy as np

cap = cv2.VideoCapture(0)
overlay = cv2.imread('Baju 2D/over-woman/full-cloth.png', cv2.IMREAD_UNCHANGED)
overlay_rgb = overlay[..., :3]
alpha_mask  = overlay[..., 3] / 255.0
h, w = overlay.shape[:2]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fh, fw = frame.shape[:2]
    x, y = fw - w, fh - h

    # Make sure these are in the order [rows, cols]
    roi = frame[y:y+h, x:x+w]

    # Debug print—both should be (33,269)
    # print("ROI:", roi.shape, "Mask:", alpha_mask.shape)

    for c in range(3):
        roi[..., c] = alpha_mask * overlay_rgb[..., c] + (1 - alpha_mask) * roi[..., c]

    frame[y:y+h, x:x+w] = roi
    cv2.imshow('Overlay', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
