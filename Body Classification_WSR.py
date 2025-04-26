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
import cv2
import mediapipe as mp
import numpy as np
import math

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
SCALE_FACTOR_SHOULDER = 0.65

def calculate_whtr(shoulder_cm, height_cm):
    return shoulder_cm / height_cm if height_cm > 0 else 0

def classify_body_type(whtr):
    if whtr < 0.62:
        return "Underweight"
    elif 0.62 <= whtr <= 0.72:
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

cap = cv2.VideoCapture(0)

cv2.namedWindow("Virtual Try-On", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Virtual Try-On", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

whtr_values = []

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

        nose_pixel_y = landmarks[mp_pose.PoseLandmark.NOSE].y * h
        left_heel_pixel_y = landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y * h
        right_heel_pixel_y = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y * h

        shoulder_pixels = abs(left_shoulder_pixel_x - right_shoulder_pixel_x)

        avg_heel_y = (left_heel_pixel_y + right_heel_pixel_y) / 2.0
        height_pixels = abs(nose_pixel_y - avg_heel_y)

        shoulder_cm = math.pi * shoulder_pixels * SCALE_FACTOR_SHOULDER * 0.63
        height_cm = height_pixels * SCALE_FACTOR_HEIGHT

        print(f"Shoulder: {shoulder_cm:.2f} cm, Height: {height_cm:.2f} cm")

        whtr = calculate_whtr(shoulder_cm, height_cm)
        whtr_values.append(whtr)

        if len(whtr_values) > 10:
            whtr_values.pop(0)

        body_type = stabilize_classification(whtr_values)

        cv2.putText(image, f"Shoulder Type: {body_type}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, f"Shoulder/Height Ratio: {whtr:.2f}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, f"Height: {height_cm:.2f} cm", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, f"Shoulder: {shoulder_cm:.2f} cm", (10, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Virtual Try-On", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
