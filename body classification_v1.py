import cv2
import mediapipe as mp
import numpy as np
import math
import torch

# Camera calibration data (same as before)
camera_matrix = np.array([
    [1.37048527e+03, 0.00000000e+00, 9.61743100e+02],
    [0.00000000e+00, 1.37063999e+03, 5.43861352e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

dist_coeffs = np.array([[ 1.57437343e-01, -8.92899556e-01, -4.33763266e-03, 
                         -5.90548476e-04,  1.02831667e+00 ]])

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Load YOLOv5 (example using Torch Hub; adjust for your YOLO version)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5  # confidence threshold
model.iou = 0.45  # IOU threshold

SCALE_FACTOR = 0.6

def calculate_whtr(waist_cm, height_cm):
    """Calculate WHtR given waist in cm and height in cm."""
    return waist_cm / height_cm if height_cm > 0 else 0

def classify_body_type(whtr):
    if whtr < 0.4:
        return "Underweight"
    elif 0.4 <= whtr <= 0.46:
        return "Ideal"
    else:
        return "Overweight"

def stabilize_classification(whtr_values, threshold=0.02):
    avg_whtr = np.mean(whtr_values)
    body_type = classify_body_type(avg_whtr)
    
    if len(whtr_values) > 1:
        if abs(whtr_values[-1] - avg_whtr) < threshold:
            return body_type
    return body_type

cap = cv2.VideoCapture(2)  # Choose the correct camera index or your video file

# Create a resizable window, then set it to 1080×1920
cv2.namedWindow("Virtual Try-On", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Virtual Try-On", 1080, 1920)

whtr_values = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame (optional)
    frame = cv2.flip(frame, 1)

    # Undistort the frame using camera calibration
    frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # -- (1) YOLO detection to get person's bounding box --
    person_height_pixels = None
    results = model(frame_undistorted)
    for *bbox, conf, cls in results.xyxy[0]:
        if int(cls) == 0:  # "person" class
            x1, y1, x2, y2 = bbox
            person_height_pixels = (y2 - y1)
            break

    # Use MediaPipe Pose for waist measurement
    image_rgb = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results_mediapipe = pose.process(image_rgb)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]

    if results_mediapipe.pose_landmarks:
        mp_drawing.draw_landmarks(
            image_bgr, results_mediapipe.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        landmarks = results_mediapipe.pose_landmarks.landmark

        # Calculate waist in pixels (hip-to-hip distance)
        left_hip_x  = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x  * w
        right_hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w
        waist_pixels = abs(left_hip_x - right_hip_x)
        waist_cm = math.pi * waist_pixels * SCALE_FACTOR

        # Use YOLO bounding-box height if available; otherwise fallback
        if person_height_pixels is not None:
            height_cm = person_height_pixels * SCALE_FACTOR
        else:
            # Fallback to nose-to-heel if YOLO fails
            nose_y      = landmarks[mp_pose.PoseLandmark.NOSE].y      * h
            left_heel_y = landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y * h
            right_heel_y= landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y* h
            avg_heel_y  = (left_heel_y + right_heel_y) / 2.0
            height_pixels = abs(nose_y - avg_heel_y)
            height_cm     = height_pixels * SCALE_FACTOR

        # Compute WHtR
        whtr = calculate_whtr(waist_cm, height_cm)
        whtr_values.append(whtr)
        if len(whtr_values) > 10:
            whtr_values.pop(0)

        # Stabilize classification
        body_type = stabilize_classification(whtr_values)

        # Display text
        cv2.putText(image_bgr, f"Body Type: {body_type}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(image_bgr, f"WHtR: {whtr:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(image_bgr, f"Height: {height_cm:.2f} cm", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(image_bgr, f"Waist: {waist_cm:.2f} cm", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Virtual Try-On", image_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
