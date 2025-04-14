import cv2
import mediapipe as mp
import numpy as np

# ----- Load Garment Images -----
# Make sure your images have an alpha channel for transparency if needed.
shirt = cv2.imread("Baju 2D/polo.png", cv2.IMREAD_UNCHANGED)
if shirt is None:
    raise FileNotFoundError("Shirt image not found. Check the path.")
pants = cv2.imread("Baju 2D/anklePants.png", cv2.IMREAD_UNCHANGED)
if pants is None:
    raise FileNotFoundError("Pants image not found. Check the path.")

# ----- Initialize MediaPipe Pose -----
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Optional scaling factors to adjust garment size relative to detected landmarks.
shirt_scale = 0.55
pants_scale = 2.0
offset_y = -50  

# ----- Define a helper function for overlaying images using alpha channels -----
def overlay_image(background, foreground):
    """Overlay foreground image on background using the alpha channel.
       Both images should be of the same size or region of interest (ROI) can be used."""
    if foreground.shape[2] == 4:
        # Separate the color and alpha channels.
        fg_rgb = foreground[:, :, :3]
        alpha_mask = foreground[:, :, 3] / 255.0
        # Blend the foreground and background.
        for c in range(3):
            background[:, :, c] = alpha_mask * fg_rgb[:, :, c] + (1 - alpha_mask) * background[:, :, c]
    else:
        background = cv2.addWeighted(background, 1, foreground, 0.5, 0)
    return background

# ----- Open the Video Capture -----
cap = cv2.VideoCapture(2)  # Adjust camera index if necessary

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror-like view.
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Process the frame using MediaPipe Pose.
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Optionally, draw the pose landmarks on the frame.
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        lm = results.pose_landmarks.landmark

        # ----- Shirt Mapping (Shoulders to Hips) -----
        # Extract shoulder landmarks.
        left_shoulder = np.array([
            lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
            lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h
        ], dtype=np.float32)
        right_shoulder = np.array([
            lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
            lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h
        ], dtype=np.float32)

        # Extract hip landmarks.
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

        # Define the destination triangle for the shirt.
        pts_dst_shirt = np.float32([
            left_shoulder + np.array([0, offset_y]),
            right_shoulder + np.array([0, offset_y]), 
            mid_ankle
            ])
        # Optional: apply scaling around the shoulder center.
        center_shirt = (left_shoulder + right_shoulder) / 2.0
        pts_dst_shirt_scaled = center_shirt + shirt_scale * (pts_dst_shirt - center_shirt)

        # Define source triangle points from the shirt image.
        # These values depend on how the garment asset is designed.
        shirt_h, shirt_w = shirt.shape[:2]
        pts_src_shirt = np.float32([
            [shirt_w * 0.3, 0],        # Corresponds to the left shoulder region in the image
            [shirt_w * 0.7, 0],        # Corresponds to the right shoulder region
            [shirt_w * 0.5, shirt_h]   # The bottom part of the shirt (to map to the hips)
        ])

        # Compute the affine transformation and warp the shirt image.
        M_shirt = cv2.getAffineTransform(pts_src_shirt, pts_dst_shirt_scaled)
        warped_shirt = cv2.warpAffine(shirt, M_shirt, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_TRANSPARENT)

        # ----- Pants Mapping (Hips to Ankles) -----
        # Use the same hip landmarks for the top of the pants.
        # Extract ankle landmarks.

        # Define the destination triangle for the pants.
        pts_dst_pants = np.float32([left_hip, right_hip, mid_ankle])
        center_pants = (left_hip + right_hip) / 2.0
        pts_dst_pants_scaled = center_pants + pants_scale * (pts_dst_pants - center_pants)

        # Define source triangle points from the pants image.
        pants_h, pants_w = pants.shape[:2]
        pts_src_pants = np.float32([
            [pants_w * 0.3, 0],        # Left hip point in the pants image
            [pants_w * 0.7, 0],        # Right hip point
            [pants_w * 0.5, pants_h]   # Bottom of the pants (to map to the ankle)
        ])

        # Compute the affine transform and warp the pants image.
        M_pants = cv2.getAffineTransform(pts_src_pants, pts_dst_pants_scaled)
        warped_pants = cv2.warpAffine(pants, M_pants, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_TRANSPARENT)

        # ----- Overlay the Garment Images onto the Frame -----
        # You can adjust the order based on which garment you want on top.
        frame = overlay_image(frame, warped_shirt)
        frame = overlay_image(frame, warped_pants)

    # Display the final augmented frame.
    cv2.imshow("Virtual Try-On", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
