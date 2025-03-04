import cv2
import mediapipe as mp
import numpy as np
import math
import torch
import open3d as o3d

camera_matrix = np.array([
    [1.37048527e+03, 0.00000000e+00, 9.61743100e+02],
    [0.00000000e+00, 1.37063999e+03, 5.43861352e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])
dist_coeffs = np.array([[1.57437343e-01, -8.92899556e-01, -4.33763266e-03, 
                         -5.90548476e-04,  1.02831667e+00 ]])

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5
model.iou = 0.45 

SCALE_FACTOR = 0.6

def calculate_whtr(waist_cm, height_cm):
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

garment_mesh = o3d.io.read_triangle_mesh("Baju_2K\\Baju_2K.glb")
garment_mesh.compute_vertex_normals()

print("Vertex colors shape:", np.asarray(garment_mesh.vertex_colors).shape)
print("Triangle UVs shape:", np.asarray(garment_mesh.triangle_uvs).shape)

vis = o3d.visualization.Visualizer()
vis.create_window(width=640, height=480, visible=False)
vis.add_geometry(garment_mesh)

render_option = vis.get_render_option()
render_option.background_color = np.array([0, 1, 0])  
view_ctl = vis.get_view_control()
cam_params = view_ctl.convert_to_pinhole_camera_parameters()


def compute_garment_transform(landmarks, w, h, garment_mesh):
    """
    Computes a transform that:
    1) Scales the garment based on waist size in pixels => cm.
    2) Rotates around Z-axis based on hip orientation.
    3) Translates in X/Y so garment "follows" the user in a basic way.
    """

    left_hip_x  = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w
    left_hip_y  = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h
    right_hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w
    right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h

    waist_pixels = abs(left_hip_x - right_hip_x)
    waist_cm = math.pi * waist_pixels * SCALE_FACTOR

    bbox = garment_mesh.get_axis_aligned_bounding_box()
    mesh_width = bbox.get_extent()[0]
    if mesh_width == 0:
        return np.eye(4)

    scale_factor = waist_cm / mesh_width

    dx = right_hip_x - left_hip_x
    dy = right_hip_y - left_hip_y
    angle_z = math.atan2(dy, dx) 

    Rz = np.eye(4)
    cosz, sinz = math.cos(angle_z), math.sin(angle_z)
    Rz[0, 0] =  cosz
    Rz[0, 1] = -sinz
    Rz[1, 0] =  sinz
    Rz[1, 1] =  cosz


    center = bbox.get_center()
    T1 = np.eye(4)
    T1[0, 3] = -center[0]
    T1[1, 3] = -center[1]
    T1[2, 3] = -center[2]

    S = np.eye(4)
    S[0, 0] = scale_factor
    S[1, 1] = scale_factor
    S[2, 2] = scale_factor

    mid_hip_x = (left_hip_x + right_hip_x) / 2.0
    mid_hip_y = (left_hip_y + right_hip_y) / 2.0
   
    factor_2d_to_3d = 1.5

    T2 = np.eye(4)
    T2[0, 3] = (mid_hip_x - (w/2)) * factor_2d_to_3d
    T2[1, 3] = (mid_hip_y - (h/2)) * factor_2d_to_3d
    T2[2, 3] = -1500.0  

    transform = T2 @ Rz @ S @ T1
    return transform

def apply_transform_to_garment(mesh, transform):
    mesh_transformed = o3d.geometry.TriangleMesh(mesh)
    mesh_transformed.transform(transform)
    return mesh_transformed

def overlay_greenscreen(bg_frame, fg_frame):
    """
    Overlays fg_frame (with green background) onto bg_frame.
    Uses basic color keying to remove the green.
    """
    hsv_fg = cv2.cvtColor(fg_frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([50, 150, 50])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv_fg, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)

    fg_img = cv2.bitwise_and(fg_frame, fg_frame, mask=mask_inv)
    bg_img = cv2.bitwise_and(bg_frame, bg_frame, mask=mask)
    return cv2.add(bg_img, fg_img)


cap = cv2.VideoCapture(1) 
cv2.namedWindow("Virtual Try-On", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Virtual Try-On", 1080, 1920)

whtr_values = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally (mirror-like)
    frame = cv2.flip(frame, 1)

    # Undistort
    frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # Person detection with YOLO
    person_height_pixels = None
    results_yolo = model(frame_undistorted)

    for *bbox, conf, cls in results_yolo.xyxy[0]:
        if int(cls) == 0:  # 'person' class
            x1, y1, x2, y2 = bbox
            person_height_pixels = (y2 - y1)
            break

    # MediaPipe Pose
    image_rgb = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2RGB)
    results_mediapipe = pose.process(image_rgb)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]

    if results_mediapipe.pose_landmarks:
        mp_drawing.draw_landmarks(
            image_bgr, results_mediapipe.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
        landmarks = results_mediapipe.pose_landmarks.landmark

        # Calculate waist size
        left_hip_x  = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x  * w
        right_hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w
        waist_pixels = abs(left_hip_x - right_hip_x)
        waist_cm = math.pi * waist_pixels * SCALE_FACTOR

        # Estimate height (using YOLO or fallback to nose→heels)
        if person_height_pixels is not None:
            height_cm = person_height_pixels * SCALE_FACTOR
        else:
            nose_y      = landmarks[mp_pose.PoseLandmark.NOSE].y       * h
            left_heel_y = landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y  * h
            right_heel_y= landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y * h
            avg_heel_y  = (left_heel_y + right_heel_y) / 2.0
            height_pixels = abs(nose_y - avg_heel_y)
            height_cm = height_pixels * SCALE_FACTOR

        # Compute WHtR
        whtr = calculate_whtr(waist_cm, height_cm)
        whtr_values.append(whtr)
        if len(whtr_values) > 10:
            whtr_values.pop(0)
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

        # Compute new garment transform each frame
        transform = compute_garment_transform(landmarks, w, h, garment_mesh)
        garment_transformed = apply_transform_to_garment(garment_mesh, transform)

        # Update offscreen visualizer
        vis.clear_geometries()
        vis.add_geometry(garment_transformed)
        vis.poll_events()
        vis.update_renderer()

        # Capture rendered garment as an image
        render = vis.capture_screen_float_buffer(do_render=True)
        render_np = (np.asarray(render) * 255).astype(np.uint8)
        render_bgr = cv2.cvtColor(render_np, cv2.COLOR_RGB2BGR)

        # Greenscreen overlay (assuming the garment image is smaller or same size)
        overlay_h, overlay_w = render_bgr.shape[:2]
        if overlay_h <= image_bgr.shape[0] and overlay_w <= image_bgr.shape[1]:
            roi = image_bgr[0:overlay_h, 0:overlay_w]
            result_roi = overlay_greenscreen(roi, render_bgr)
            image_bgr[0:overlay_h, 0:overlay_w] = result_roi

    cv2.imshow("Virtual Try-On", image_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
vis.destroy_window()
