import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector

# Initialize video capture and pose detector
cap = cv2.VideoCapture(1)
detector = PoseDetector()

# Read one frame to get dimensions for camera calibration
ret, frame = cap.read()
h, w, _ = frame.shape
focal_length = w  # rough approximation
center = (w / 2, h / 2)
camera_matrix = np.array([[focal_length, 0, center[0]],
                          [0, focal_length, center[1]],
                          [0, 0, 1]], dtype="double")
dist_coeffs = np.zeros((4, 1))  # assuming no lens distortion

# Define 3D model points for at least 6 correspondences.
# These model points are defined in an arbitrary coordinate system.
# Adjust these values to suit your calibration and 3D model.
model_points = np.array([
    (0.0, 0.0, 0.0),       # Nose (landmark 0)
    (-10.0, 0.0, -10.0),   # Left Shoulder (landmark 11)
    (10.0, 0.0, -10.0),    # Right Shoulder (landmark 12)
    (-15.0, -10.0, -20.0), # Left Elbow (landmark 13)
    (15.0, -10.0, -20.0),  # Right Elbow (landmark 14)
    (0.0, -30.0, -10.0)    # Hip Center (average of left (23) and right hip (24))
])

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect pose and landmarks on the image
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, draw=False)
    
    # Ensure we have enough landmarks (e.g. Mediapipe Pose returns 33)
    if lmList and len(lmList) >= 25:
        try:
            # Extract required 2D image points from landmarks
            nose = lmList[0][1:3]            # Landmark 0: Nose
            left_shoulder = lmList[11][1:3]    # Landmark 11: Left Shoulder
            right_shoulder = lmList[12][1:3]   # Landmark 12: Right Shoulder
            left_elbow = lmList[13][1:3]       # Landmark 13: Left Elbow
            right_elbow = lmList[14][1:3]      # Landmark 14: Right Elbow
            left_hip = lmList[23][1:3]         # Landmark 23: Left Hip
            right_hip = lmList[24][1:3]        # Landmark 24: Right Hip
        except IndexError:
            continue
        
        # Compute the hip center as the average of left and right hips
        hip_center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
        
        # Assemble the six 2D image points in the same order as model_points
        image_points = np.array([
            nose,
            left_shoulder,
            right_shoulder,
            left_elbow,
            right_elbow,
            hip_center
        ], dtype="double")
        
        # Perform pose estimation using solvePnP. Although many solvePnP methods
        # work with 4+ points, using 6 points is recommended for DLT-based approaches.
        success_pnp, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success_pnp:
            # Define a simple cube as our 3D model relative to the pose (attached at the nose)
            cube_size = 10.0  # Adjust as needed
            cube_points = np.array([
                (-cube_size, -cube_size, 0),
                (-cube_size, cube_size, 0),
                (cube_size, cube_size, 0),
                (cube_size, -cube_size, 0),
                (-cube_size, -cube_size, -cube_size*2),
                (-cube_size, cube_size, -cube_size*2),
                (cube_size, cube_size, -cube_size*2),
                (cube_size, -cube_size, -cube_size*2)
            ])
            
            # Project the 3D cube points onto the 2D image plane
            projected_points, _ = cv2.projectPoints(
                cube_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs
            )
            projected_points = projected_points.reshape(-1, 2).astype(int)
            
            # Draw the cube edges on the image
            cv2.polylines(img, [projected_points[:4]], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.polylines(img, [projected_points[4:]], isClosed=True, color=(0, 255, 0), thickness=2)
            for i in range(4):
                cv2.line(img, tuple(projected_points[i]), tuple(projected_points[i+4]), color=(0, 0, 255), thickness=2)
    
    cv2.imshow("3D Model on Pose", img)
    key = cv2.waitKey(1)
    if key == 27:  # Exit on 'Esc' key
        break

cap.release()
cv2.destroyAllWindows()
