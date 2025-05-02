import cv2
import mediapipe as mp
import numpy as np


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

shirt_scale = 0.8
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

    # Display the final augmented frame.
    cv2.imshow("Virtual Try-On", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()