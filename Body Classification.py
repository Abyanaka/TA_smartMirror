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

# --- (1) Define scale factor based on known reference ---
# For example, an object of 165 cm was ~294.46 px tall at the same distance
SCALE_FACTOR = 165.0 / 294.46  # cm/px

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
    # Apply threshold to prevent frequent changes
    if len(whtr_values) > 1:
        if abs(whtr_values[-1] - avg_whtr) < threshold:
            return body_type
    return body_type

cap = cv2.VideoCapture(2)  # Webcam or your video source

# Create a fullscreen window
cv2.namedWindow("Virtual Try-On", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Virtual Try-On", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

whtr_values = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame (optional)
    frame = cv2.flip(frame, 1)

    # Undistort the frame using camera calibration
    frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # Convert to RGB for MediaPipe
    image = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)

    # Convert back to BGR for drawing
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Get image dimensions (height, width)
    h, w = image.shape[:2]

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # Extract landmarks
        landmarks = results.pose_landmarks.landmark

        # Convert normalized coordinates to pixel coordinates
        left_hip_pixel_x  = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x  * w
        right_hip_pixel_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w

        nose_pixel_y      = landmarks[mp_pose.PoseLandmark.NOSE].y      * h
        left_heel_pixel_y = landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y * h
        right_heel_pixel_y= landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y* h

        # Calculate waist in pixels (horizontal distance)
        waist_pixels = (abs(left_hip_pixel_x - right_hip_pixel_x))
        # Calculate height in pixels (vertical distance)
        avg_heel_y = (left_heel_pixel_y + right_heel_pixel_y) / 2.0
        height_pixels = abs(nose_pixel_y - avg_heel_y)

        # --- (2) Convert pixels to centimeters using SCALE_FACTOR ---
        waist_cm  = math.pi * waist_pixels  * SCALE_FACTOR
        height_cm = height_pixels * SCALE_FACTOR

        # Print waist and height (in cm) to terminal
        print(f"Waist: {waist_cm:.2f} cm, Height: {height_cm:.2f} cm")

        # --- (3) Compute WHtR using real-world units ---
        whtr = calculate_whtr(waist_cm, height_cm)
        whtr_values.append(whtr)

        # Keep only the last 10
        if len(whtr_values) > 10:
            whtr_values.pop(0)

        # Stabilize classification
        body_type = stabilize_classification(whtr_values)

        # Display body type and WHtR
        cv2.putText(image, f"Body Type: {body_type}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(image, f"WHtR: {whtr:.2f}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(image, f"Height: {height_cm:.2f}", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(image, f"Waist {waist_cm:.2f}", (10, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Virtual Try-On", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
