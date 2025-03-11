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
dist_coeffs = np.array([[ 1.57437343e-01, -8.92899556e-01, -4.33763266e-03, 
                         -5.90548476e-04,  1.02831667e+00 ]])

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5  # confidence threshold
model.iou = 0.45  # IOU threshold

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


garment_mesh = o3d.io.read_triangle_mesh("Baju_2K\Baju_2K.glb")
garment_mesh.compute_vertex_normals()

vis = o3d.visualization.Visualizer()
vis.create_window(width=640, height=480, visible=False)
vis.add_geometry(garment_mesh)

render_option = vis.get_render_option()
render_option.background_color = np.array([0, 1, 0])

view_ctl = vis.get_view_control()
cam_params = view_ctl.convert_to_pinhole_camera_parameters() 

def compute_garment_transform(landmarks, w, h, garment_mesh):
    
    left_hip_x  = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w
    right_hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w
    waist_pixels = abs(left_hip_x - right_hip_x)
    waist_cm = math.pi * waist_pixels * SCALE_FACTOR

    bbox = garment_mesh.get_axis_aligned_bounding_box()
    mesh_width = bbox.get_extent()[0]  
    if mesh_width == 0:
        return np.eye(4)

    scale_factor = (waist_cm / mesh_width)*2

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
    T2[2, 3] = -1500.0

    return T2 @ S @ T1

def apply_transform_to_garment(mesh, transform):
    mesh_transformed = o3d.geometry.TriangleMesh(mesh)
    mesh_transformed.transform(transform)
    return mesh_transformed

def overlay_greenscreen(bg_frame, fg_frame):
    hsv_fg = cv2.cvtColor(fg_frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([50, 150, 50])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv_fg, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)


    fg_img = cv2.bitwise_and(fg_frame, fg_frame, mask=mask_inv)
    # Keep only the region from bg where mask=green
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

    frame = cv2.flip(frame, 1)

    frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

    person_height_pixels = None
    results_yolo = model(frame_undistorted)
    for *bbox, conf, cls in results_yolo.xyxy[0]:
        if int(cls) == 0: 
            x1, y1, x2, y2 = bbox
            person_height_pixels = (y2 - y1)
            break

    image_rgb = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2RGB)
    results_mediapipe = pose.process(image_rgb)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]

    if results_mediapipe.pose_landmarks:
        mp_drawing.draw_landmarks(
            image_bgr, results_mediapipe.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
        landmarks = results_mediapipe.pose_landmarks.landmark

        left_hip_x  = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x  * w
        right_hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w
        waist_pixels = abs(left_hip_x - right_hip_x)
        waist_cm = math.pi * waist_pixels * SCALE_FACTOR

        if person_height_pixels is not None:
            height_cm = person_height_pixels * SCALE_FACTOR
        else:
            nose_y      = landmarks[mp_pose.PoseLandmark.NOSE].y      * h
            left_heel_y = landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y * h
            right_heel_y= landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y* h
            avg_heel_y  = (left_heel_y + right_heel_y) / 2.0
            height_pixels = abs(nose_y - avg_heel_y)
            height_cm = height_pixels * SCALE_FACTOR

        whtr = calculate_whtr(waist_cm, height_cm)
        whtr_values.append(whtr)
        if len(whtr_values) > 10:
            whtr_values.pop(0)
        body_type = stabilize_classification(whtr_values)

        cv2.putText(image_bgr, f"Body Type: {body_type}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(image_bgr, f"WHtR: {whtr:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(image_bgr, f"Height: {height_cm:.2f} cm", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(image_bgr, f"Waist: {waist_cm:.2f} cm", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        transform = compute_garment_transform(landmarks, w, h, garment_mesh)
        garment_transformed = apply_transform_to_garment(garment_mesh, transform)


        vis.clear_geometries()
        vis.add_geometry(garment_transformed)
        vis.poll_events()
        vis.update_renderer()

        render = vis.capture_screen_float_buffer(do_render=True)
        render_np = (np.asarray(render) * 255).astype(np.uint8)
        render_bgr = cv2.cvtColor(render_np, cv2.COLOR_RGB2BGR)

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
