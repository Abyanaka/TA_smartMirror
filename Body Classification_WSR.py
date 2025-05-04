import cv2
import mediapipe as mp
import numpy as np
import math

# Camera calibration data (same as before)
camera_matrix = np.array([
    [1.37048527e+03, 0.00000000e+00, 9.61743100e+02],
    [0.00000000e+00, 1.37063999e+03, 5.43861352e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

dist_coeffs = np.array([[ 1.57437343e-01, -8.92899556e-01, -4.33763266e-03, 
                         -5.90548476e-04,  1.02831667e+00 ]])


# Camera calibration data
camera_matrix = np.array([
    [1.37048527e+03, 0.00000000e+00, 9.61743100e+02],
    [0.00000000e+00, 1.37063999e+03, 5.43861352e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

dist_coeffs = np.array([[1.57437343e-01, -8.92899556e-01, -4.33763266e-03, 
                         -5.90548476e-04, 1.02831667e+00]])

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

SCALE_FACTOR_HEIGHT = 0.45
SCALE_FACTOR_SHOULDER = 0.38  # cm per pixel di bidang bahu
SCALE_FACTOR_WAIST    = 0.365  # anggap sama untuk horizontal torso
WAIST_INTERP_FACTOR   = 0.451   # 30% naik dari pinggul ke bahu

def calculate_wsr(shoulder_cm, waist_cm):
    return waist_cm / shoulder_cm if height_cm > 0 else 0

def classify_body_type(wsr):
    if wsr < 0.62:
        return "Underweight"
    elif 0.62 <= wsr <= 0.72:
        return "Ideal"
    else:
        return "Overweight"

def stabilize_classification(wsr_values, threshold=0.02):
    avg_wsr = np.mean(wsr_values)
    body_type = classify_body_type(avg_wsr)

    if len(wsr_values) > 1:
        if abs(wsr_values[-1] - avg_wsr) < threshold:
            return body_type
    return body_type

cap = cv2.VideoCapture(0)

# cv2.namedWindow("Virtual Try-On", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("Virtual Try-On", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

wsr_values = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

    image = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h, w = image.shape[:2]

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark

        left_shoulder_pixel_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w
        right_shoulder_pixel_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w
        shoulder_pixels = abs(left_shoulder_pixel_x - right_shoulder_pixel_x)

        left_hip_pixel_x  = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x  * w
        right_hip_pixel_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w
        lhs = left_hip_pixel_x + WAIST_INTERP_FACTOR * (left_shoulder_pixel_x - left_hip_pixel_x)
        rhs = right_hip_pixel_x + WAIST_INTERP_FACTOR * (right_shoulder_pixel_x - right_hip_pixel_x)
        waist_pixels = abs(lhs - rhs)

        nose_pixel_y = landmarks[mp_pose.PoseLandmark.NOSE].y * h
        left_heel_pixel_y = landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y * h
        right_heel_pixel_y = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y * h
        avg_heel_y = (left_heel_pixel_y + right_heel_pixel_y) / 2.0
        height_pixels = abs(nose_pixel_y - avg_heel_y)



        shoulder_cm = math.pi * shoulder_pixels * SCALE_FACTOR_SHOULDER
        waist_cm  = math.pi * waist_pixels  * SCALE_FACTOR_WAIST
        height_cm = height_pixels * SCALE_FACTOR_HEIGHT

        print(f"Shoulder: {shoulder_cm:.2f} cm, Height: {height_cm:.2f} cm")

        wsr = calculate_wsr(shoulder_cm, waist_cm)
        wsr_values.append(wsr)

        if len(wsr_values) > 10:
            wsr_values.pop(0)

        body_type = stabilize_classification(wsr_values)

        cv2.putText(image, f"Shoulder Type: {body_type}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, f"Shoulder/Height Ratio: {wsr:.2f}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, f"Height: {height_cm:.2f} cm", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, f"Shoulder: {shoulder_cm:.2f} cm", (10, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, f"Waist {waist_cm:.2f}", (10, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", 1200, 2000)
    cv2.imshow("Frame", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
