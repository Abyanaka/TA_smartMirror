import cv2
import textwrap
import mediapipe as mp
import numpy as np

icon = cv2.imread("Baju 2D/ideal-man/polo-full.png", cv2.IMREAD_UNCHANGED)
if icon is None:
    raise FileNotFoundError("Icon not found")


shirt = cv2.imread("Baju 2D\ideal-man\pololo.png", cv2.IMREAD_UNCHANGED)
if shirt is None:
    raise FileNotFoundError("Shirt image not found. Check the path.")

pants = cv2.imread("Baju 2D\ideal-man/anklePants.png", cv2.IMREAD_UNCHANGED)
if pants is None:
    raise FileNotFoundError("Pants image not found. Check the path.")

right_sleeve = cv2.imread("Baju 2D\ideal-man\pololo-arm-right.png", cv2.IMREAD_UNCHANGED)
if right_sleeve is None:
    raise FileNotFoundError("right_sleeve image not found.")

left_sleeve = cv2.imread("Baju 2D\ideal-man\pololo-arm-left.png", cv2.IMREAD_UNCHANGED)
if left_sleeve is None:
    raise FileNotFoundError("left_sleeve image not found.")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

icon_scale = 0.15
shirt_scale = 0.8
pants_scale = 1.0
sleeve_scale = 1.0

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
new_w = 100
new_h = 120
icon = cv2.resize(icon, (new_w, new_h), interpolation=cv2.INTER_AREA)


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

        left_hip = np.array([
            lm[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
            lm[mp_pose.PoseLandmark.LEFT_HIP.value].y * h
        ], dtype=np.float32)
        right_hip = np.array([
            lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
            lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h
        ], dtype=np.float32)
        mid_hip = (left_hip + right_hip) / 2.0

        left_ankle = np.array([
            lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
            lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h
        ], dtype=np.float32)
        right_ankle = np.array([
            lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
            lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h
        ], dtype=np.float32)
        mid_ankle = (left_ankle + right_ankle) / 2.0

        right_elbow = np.array([
            lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
            lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h
        ], dtype=np.float32)
        right_mid = (right_shoulder + right_elbow) / 2


        left_elbow = np.array([
            lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
            lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h
        ], dtype=np.float32)
        left_mid = (left_shoulder + left_elbow) / 2


        ######################################### SHIRT #############################################
        offset_shirt = -50  
        pts_dst_shirt = np.float32([
            left_shoulder + np.array([0, offset_shirt]),
            right_shoulder + np.array([0, offset_shirt]), 
            right_hip,
            left_hip
        ])
        center_shirt = (left_shoulder + right_shoulder) / 2.0
        pts_dst_shirt_scaled = center_shirt + shirt_scale * (pts_dst_shirt - center_shirt)

        shirt_h, shirt_w = shirt.shape[:2]
        pts_src_shirt = np.float32([
            [shirt_w * 0.2, 0],          # kiri atas
            [shirt_w * 0.8, 0],          # kanan atas
            [shirt_w * 0.7, shirt_h],    # kanan bawah
            [shirt_w * 0.3, shirt_h]     # kiri bawah
        ])

        M_shirt = cv2.getPerspectiveTransform(pts_src_shirt, pts_dst_shirt_scaled)
        warped_shirt = cv2.warpPerspective(shirt, M_shirt, (w, h),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_TRANSPARENT)
        
        ################################# SLEEVES ###################################################
        pts_dst_right_sleeve = np.float32([
            right_shoulder+ np.array([-20, -20]), 
            right_shoulder + np.array([0, -20]), 
            right_elbow
        ])
        pts_dst_right_sleeve_scaled = right_mid + sleeve_scale * (pts_dst_right_sleeve - right_mid)

        right_sleeve_h, right_sleeve_w = right_sleeve.shape[:2]
        pts_src_right_sleeve = np.float32([
            [right_sleeve_w * 0.3, 0],
            [right_sleeve_w * 0.7, 0],
            [right_sleeve_w * 0.5, right_sleeve_h]
        ])

        M_right_sleeve = cv2.getAffineTransform(pts_src_right_sleeve, pts_dst_right_sleeve_scaled)
        warped_right_sleeve = cv2.warpAffine(right_sleeve, M_right_sleeve, (w, h), 
                                             flags=cv2.INTER_LINEAR, 
                                             borderMode=cv2.BORDER_TRANSPARENT)
        

        pts_dst_left_sleeve = np.float32([
            left_shoulder + np.array([0, -20]), 
            left_shoulder + np.array([20, -20]), 
            left_elbow
            ])
        pts_dst_left_sleeve_scaled = left_mid + sleeve_scale * (pts_dst_left_sleeve - left_mid)

        left_sleeve_h, left_sleeve_w = left_sleeve.shape[:2]
        pts_src_left_sleeve = np.float32([
            [left_sleeve_w * 0.3, 0],
            [left_sleeve_w * 0.7, 0],
            [left_sleeve_w * 0.5, left_sleeve_h]
        ])

        M_left_sleeve = cv2.getAffineTransform(pts_src_left_sleeve, pts_dst_left_sleeve_scaled)
        warped_left_sleeve = cv2.warpAffine(left_sleeve, M_left_sleeve, (w, h), 
                                             flags=cv2.INTER_LINEAR, 
                                             borderMode=cv2.BORDER_TRANSPARENT)

        ############################################### PANTS ######################################
        hip_offset_y = -40
        pts_dst_pants = np.float32([
            left_hip + np.array([0, hip_offset_y]),
            right_hip + np.array([0, hip_offset_y]),
            mid_ankle
        ])  
        center_pants = (left_hip + right_hip) / 2.0
        pts_dst_pants_scaled = center_pants + pants_scale * (pts_dst_pants - center_pants)

        
        pants_h, pants_w = pants.shape[:2]
        pts_src_pants = np.float32([
            [pants_w * 0.4, 0],        
            [pants_w * 0.6, 0],       
            [pants_w * 0.5, pants_h]   
        ])

        M_pants = cv2.getAffineTransform(pts_src_pants, pts_dst_pants_scaled)
        warped_pants = cv2.warpAffine(pants, M_pants, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_TRANSPARENT)

        frame = overlay_image(frame, warped_pants)

        frame = overlay_image(frame, warped_right_sleeve)
        frame = overlay_image(frame, warped_left_sleeve)

        frame = overlay_image(frame, warped_shirt)

    caption = ("Kamu dengan tubuh ideal, hampir semua jenis pakaian dapat digunakan, asalkan tetap memperhatikan proporsi tubuh. Pakaian dengan potongan yang pas di badan (well-fitted) sangat dianjurkan. Penggunaan celana slim-fit dapat membantu menonjolkan bentuk tubuh yang proporsional.")
    
    x, y = 10, 400
    max_chars_per_line = 75
    textWrap(frame, caption, x, y, max_chars_per_line)

    # Display the final augmented frame.
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", 1200, 2000)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()