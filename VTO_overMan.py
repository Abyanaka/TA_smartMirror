import cv2
import mediapipe as mp
import numpy as np


shirt = cv2.imread("Baju 2D/over-man/dark-denim-mid part.png", cv2.IMREAD_UNCHANGED)
if shirt is None:
    raise FileNotFoundError("Shirt image not found. Check the path.")

pants = cv2.imread("Baju 2D/over-man/straight-jeans.png", cv2.IMREAD_UNCHANGED)
if pants is None:
    raise FileNotFoundError("Pants image not found. Check the path.")

right_sleeve = cv2.imread("Baju 2D/over-man/dark-denim-arm.png", cv2.IMREAD_UNCHANGED)
if right_sleeve is None:
    raise FileNotFoundError("right_sleeve image not found.")

left_sleeve = cv2.imread("Baju 2D/over-man/dark-denim-arm-1.png", cv2.IMREAD_UNCHANGED)
if left_sleeve is None:
    raise FileNotFoundError("left_sleeve image not found.")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

shirt_scale = 0.9
pants_scale = 1.0
sleeve_scale = 1.0



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

def sleeves(frame, sleeve_img, shoulder, elbow, wrist, scale=1.0, offset=np.array([0, 0])):
    h_sleeve, w_sleeve = sleeve_img.shape[:2]
    half_h = h_sleeve // 2

    pts_src_upper = np.float32([
        [0, 0],
        [w_sleeve, 0],
        [w_sleeve / 2, half_h]
    ])
    pts_src_lower = np.float32([
        [0, 0],
        [w_sleeve, 0],
        [w_sleeve / 2, half_h]
    ])

    shoulder = (shoulder + offset).astype(np.float32)
    elbow = (elbow + offset).astype(np.float32)
    wrist = (wrist + offset).astype(np.float32)

    # Create direction vectors
    dir_upper = elbow - shoulder
    dir_upper /= np.linalg.norm(dir_upper)
    perp_upper = np.array([-dir_upper[1], dir_upper[0]]) * 40  # width = 80

    dir_lower = wrist - elbow
    dir_lower /= np.linalg.norm(dir_lower)
    perp_lower = np.array([-dir_lower[1], dir_lower[0]]) * 40  # width = 80

    # Build triangles dynamically
    pts_dst_upper = np.float32([
        shoulder - perp_upper,
        shoulder + perp_upper,
        elbow
    ])

    pts_dst_lower = np.float32([
        elbow - perp_lower,
        elbow + perp_lower,
        wrist
    ])

    upper_half = sleeve_img[0:half_h]
    lower_half = sleeve_img[half_h:]

    M_upper = cv2.getAffineTransform(pts_src_upper, pts_dst_upper)
    warped_upper = cv2.warpAffine(upper_half, M_upper, (frame.shape[1], frame.shape[0]),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    M_lower = cv2.getAffineTransform(pts_src_lower, pts_dst_lower)
    warped_lower = cv2.warpAffine(lower_half, M_lower, (frame.shape[1], frame.shape[0]),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    frame = overlay_image(frame, warped_upper)
    frame = overlay_image(frame, warped_lower)
    return frame



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

        right_wrist = np.array([
            lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
            lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h
        ], dtype=np.float32)
        left_wrist = np.array([
            lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
            lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h
        ], dtype=np.float32)


        ######################################### SHIRT #############################################
        offset_shirt = -40  
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
        frame = overlay_image(frame, warped_shirt)
        frame = sleeves(frame, right_sleeve, right_shoulder, right_elbow, right_wrist, scale=1.0, offset=np.array([0, -10]))
        frame = sleeves(frame, left_sleeve, left_shoulder, left_elbow, left_wrist, scale=1.0, offset=np.array([0, -10]))


    # Display the final augmented frame.
    cv2.imshow("Virtual Try-On", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()