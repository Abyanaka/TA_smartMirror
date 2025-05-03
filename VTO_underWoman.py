import cv2
import textwrap
import mediapipe as mp
import numpy as np

icon = cv2.imread("Baju 2D/under-woman/DRESSYellow.png", cv2.IMREAD_UNCHANGED)
if icon is None:
    raise FileNotFoundError("Icon not found")

shirt = icon
if shirt is None:
    raise FileNotFoundError("Shirt image not found. Check the path.")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

icon_scale = 0.15
shirt_scale = 1.0

def roundedRect(img, top_left, bottom_right, color, radius):
    
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness=cv2.FILLED)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness=cv2.FILLED)
    
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness=cv2.FILLED)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness=cv2.FILLED)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness=cv2.FILLED)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness=cv2.FILLED)

def textWrap(frame, text, x, y, max_chars_per_line,
             font=cv2.FONT_HERSHEY_SIMPLEX, 
             font_scale=0.5,
             thickness=1, 
             color_text=(255, 255, 255), 
             color_bg=(0, 0, 0),
             padding=20, 
             alpha=0.5, 
             max_lines=5, 
             radius=20):
    

    wrapped_lines = textwrap.wrap(text, width=max_chars_per_line)
    if len(wrapped_lines) > max_lines:
        wrapped_lines = wrapped_lines[:max_lines]
    
    line_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in wrapped_lines]
    text_width = max(size[0] for size in line_sizes)
    line_height = line_sizes[0][1] 
    
    rect_x1 = x - padding
    rect_y1 = y - line_height - padding
    rect_x2 = x + text_width + padding
    rect_y2 = y + len(wrapped_lines) * line_height + padding

    overlay = frame.copy()

    roundedRect(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), color_bg, radius)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    for i, line in enumerate(wrapped_lines):
        line_y = y + i * (line_height + 5)
        cv2.putText(frame, line, (x, line_y), font, font_scale, color_text, thickness)

def overlay_image(background, foreground):
   
    if foreground.shape[2] == 4:

        fg_rgb = foreground[:, :, :3]
        alpha_mask = foreground[:, :, 3] / 255.0

        for c in range(3):
            background[:, :, c] = alpha_mask * fg_rgb[:, :, c] + (1 - alpha_mask) * background[:, :, c]
    else:
        background = cv2.addWeighted(background, 1, foreground, 0.5, 0)
    return background

cap = cv2.VideoCapture(0)

orig_h, orig_w = icon.shape[:2]
new_h = 100
new_w = 100
icon = cv2.resize(icon, (new_h, new_w), interpolation=cv2.INTER_AREA)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    x_offset = 500
    y_offset = 10

    bg_alpha = 0.3
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x_offset, y_offset),
        (x_offset + new_w, y_offset + new_h),
        (255, 255, 255),
        thickness=cv2.FILLED
    )

    frame = cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0)
    roi = frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
    if icon.shape[2] == 4:
        icon_rgb = icon[:, :, :3]
        alpha = icon[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (alpha * icon_rgb[:, :, c] +
                            (1 - alpha) * roi[:, :, c])
        frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = roi
    else:
        frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = icon

    if results.pose_landmarks:

        # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        lm = results.pose_landmarks.landmark

        left_shoulder = np.array([
            lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
            lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h
        ], dtype=np.float32)
        right_shoulder = np.array([
            lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
            lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h
        ], dtype=np.float32)


        left_ankle = np.array([
            lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
            lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h
        ], dtype=np.float32)
        right_ankle = np.array([
            lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
            lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h
        ], dtype=np.float32)
        mid_ankle = (left_ankle + right_ankle) / 2.0


        ######################################### SHIRT #############################################
        offset_shirt = -50  
        pts_dst_shirt = np.float32([
            left_shoulder + np.array([0, offset_shirt]),
            right_shoulder + np.array([0, offset_shirt]), 
            mid_ankle
        ])
        center_shirt = (left_shoulder + right_shoulder) / 2.0
        pts_dst_shirt_scaled = center_shirt + shirt_scale * (pts_dst_shirt - center_shirt)

        shirt_h, shirt_w = shirt.shape[:2]
        pts_src_shirt = np.float32([
            [shirt_w * 0.3, 0],              
            [shirt_w * 0.7, 0],              
            [shirt_w * 0.5, shirt_h]       
        ])

        M_shirt = cv2.getAffineTransform(pts_src_shirt, pts_dst_shirt_scaled)
        warped_shirt = cv2.warpAffine(shirt, M_shirt, (w, h),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_TRANSPARENT)

        frame = overlay_image(frame, warped_shirt)

    caption = ("Wanita bertubuh kurus sebaiknya memilih pakaian berwarna cerah untuk memberikan kesan lebih berisi. Pilihan pakaian berukuran sedang (medium fit) lebih dianjurkan daripada yang terlalu ketat atau longgar. Penggunaan outer seperti cardigan, jaket, atau blazer, serta teknik layering, dapat menambah volume pada penampilan. Hindari pakaian oversized atau yang terlalu ketat agar tubuh tetap terlihat proporsional.")
    
    x, y = 10, 400
    max_chars_per_line = 75
    textWrap(frame, caption, x, y, max_chars_per_line)

    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", 1200, 2000)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()