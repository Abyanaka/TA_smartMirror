import cv2
import mediapipe as mp
import numpy as np


shirt = cv2.imread("Baju 2D/under-woman/DRESSYellow.png", cv2.IMREAD_UNCHANGED)
if shirt is None:
    raise FileNotFoundError("Shirt image not found. Check the path.")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

shirt_scale = 1.0

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

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

    # Display the final augmented frame.
    cv2.imshow("Virtual Try-On", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()