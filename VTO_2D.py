import cv2
import mediapipe as mp
import numpy as np

# ----- Initialization -----
# Load the 2D garment image with an alpha channel (ensure the file has transparency)
garment = cv2.imread("Baju 2D/Group 9.png", cv2.IMREAD_UNCHANGED)
if garment is None:
    raise FileNotFoundError("Garment image not found. Check the path.")

# Define MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(2)

# Set your desired garment scaling factor (less than 1 scales down the garment)
garment_scale = 0.9

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # mirror image for natural interaction
    h, w, _ = frame.shape

    # Convert to RGB and process with MediaPipe Pose
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # (Optional) Draw landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # ----- Extract user anchor points from MediaPipe -----
        lm = results.pose_landmarks.landmark
        left_shoulder = np.array([lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                                  lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h], dtype=np.float32)
        right_shoulder = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                                   lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h], dtype=np.float32)
        # Use the midpoint between left and right hip as the bottom anchor
        left_hip = np.array([lm[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                             lm[mp_pose.PoseLandmark.LEFT_HIP.value].y * h], dtype=np.float32)
        right_hip = np.array([lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                              lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h], dtype=np.float32)
        
        bottom_center = ((left_hip + right_hip) / 2) 

        # Original user anchor points (destination points for affine transform)
        pts_dst = np.float32([left_shoulder, right_shoulder, bottom_center])
        
        # ----- Scale down the garment by moving points towards the shoulder center -----
        center_shoulder = (left_shoulder + right_shoulder) / 2
        pts_dst_scaled = center_shoulder + garment_scale * (pts_dst - center_shoulder)
        pts_dst_scaled = np.float32(pts_dst_scaled)  # ensure type is float32

        # ----- Define garment anchor points (source points) -----
        garment_h, garment_w = garment.shape[:2]
        pts_src = np.float32([
            [garment_w * 0.3, garment_h * 0.3],   # left shoulder on garment
            [garment_w * 0.7, garment_h * 0.3],   # right shoulder on garment
            [garment_w * 0.5, garment_h * 0.9]      # bottom center of garment
        ])

        # Compute the affine transformation matrix from garment to scaled user points
        M = cv2.getAffineTransform(pts_src, pts_dst_scaled)
        
        # Warp the garment image so it fits onto the user
        warped_garment = cv2.warpAffine(garment, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        
        # ----- Overlay the warped garment onto the frame using its alpha channel -----
        if warped_garment.shape[2] == 4:
            # Separate the BGR and alpha channels
            garment_rgb = warped_garment[:, :, :3]
            alpha_mask = warped_garment[:, :, 3] / 255.0
            # Blend the garment with the frame based on the alpha channel
            for c in range(3):
                frame[:, :, c] = (alpha_mask * garment_rgb[:, :, c] +
                                  (1 - alpha_mask) * frame[:, :, c])
        else:
            # If no alpha channel, do a simple overlay with 50% transparency
            frame = cv2.addWeighted(frame, 1, warped_garment, 0.5, 0)

    cv2.imshow("2D Virtual Try-On", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
