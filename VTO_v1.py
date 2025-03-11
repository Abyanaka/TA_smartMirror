import cv2
import mediapipe as mp
import numpy as np
import math
import torch
import open3d as o3d

# 1) Set up camera calibration (if you have it).
camera_matrix = np.array([
    [1.37048527e+03, 0.00000000e+00, 9.61743100e+02],
    [0.00000000e+00, 1.37063999e+03, 5.43861352e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])
dist_coeffs = np.array([[ 1.57437343e-01, -8.92899556e-01, -4.33763266e-03, 
                         -5.90548476e-04,  1.02831667e+00 ]])

# 2) Initialize Pose and YOLO (YOLO is optional if you just want bounding boxes).
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 3) Load the 3D clothing model.
garment_mesh = o3d.io.read_triangle_mesh("Baju_2K\\Baju_2K.glb")
garment_mesh.compute_vertex_normals()

# 4) Setup an Open3D Visualizer for offscreen rendering.
vis = o3d.visualization.Visualizer()
vis.create_window(width=640, height=480, visible=False)
vis.add_geometry(garment_mesh)

# 5) Helper function to remove green background and overlay garment on the camera frame.
def overlay_greenscreen(bg_frame, fg_frame):
    hsv_fg = cv2.cvtColor(fg_frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([50, 150, 50])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv_fg, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)

    fg_img = cv2.bitwise_and(fg_frame, fg_frame, mask=mask_inv)
    bg_img = cv2.bitwise_and(bg_frame, bg_frame, mask=mask)
    return cv2.add(bg_img, fg_img)

# 6) Simple transform so the 3D model matches the user’s waist size.
def compute_garment_transform(landmarks, w, h, garment_mesh):
    # (a) waist pixels from left_hip to right_hip
    left_hip_x  = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w
    right_hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w
    waist_pixels = abs(left_hip_x - right_hip_x)

    # (b) Convert pixels -> cm (naive scale)
    SCALE_FACTOR = 0.6
    waist_cm = math.pi * waist_pixels * SCALE_FACTOR

    # (c) Get bounding box of the garment mesh
    bbox = garment_mesh.get_axis_aligned_bounding_box()
    mesh_width = bbox.get_extent()[0]

    # (d) scale factor for the garment
    if mesh_width == 0:
        return np.eye(4)
    scale_factor = waist_cm / mesh_width

    # (e) Transform: center mesh, scale, then push it “forward.”
    center = bbox.get_center()
    T1 = np.eye(4)
    T1[0, 3] = -center[0]
    T1[1, 3] = -center[1]
    T1[2, 3] = -center[2]

    S = np.eye(4)
    S[0, 0] = scale_factor
    S[1, 1] = scale_factor
    S[2, 2] = scale_factor

    T2 = np.eye(4)
    T2[2, 3] = -1500.0  # Hard-coded forward shift

    return T2 @ S @ T1

def apply_transform_to_garment(mesh, transform):
    mesh_t = o3d.geometry.TriangleMesh(mesh)
    mesh_t.transform(transform)
    return mesh_t

# 7) Main loop: read frames, detect pose, scale garment, render, overlay
cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_undist = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # MediaPipe Pose
    image_rgb = cv2.cvtColor(frame_undist, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]

    if results.pose_landmarks:
        # Draw pose
        mp.solutions.drawing_utils.draw_landmarks(
            image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
        transform = compute_garment_transform(
            results.pose_landmarks.landmark, w, h, garment_mesh
        )
        garment_transformed = apply_transform_to_garment(garment_mesh, transform)

        # Render with Open3D
        vis.clear_geometries()
        vis.add_geometry(garment_transformed)
        vis.poll_events()
        vis.update_renderer()

        render = vis.capture_screen_float_buffer(do_render=True)
        render_np = (np.asarray(render) * 255).astype(np.uint8)
        render_bgr = cv2.cvtColor(render_np, cv2.COLOR_RGB2BGR)

        # Overlay garment
        overlayed = overlay_greenscreen(image_bgr, render_bgr)
        cv2.imshow("Virtual Try-On", overlayed)
    else:
        # If no pose, just show the camera
        cv2.imshow("Virtual Try-On", image_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
vis.destroy_window()
