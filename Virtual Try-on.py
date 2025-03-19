import cv2
import mediapipe as mp
import numpy as np
import open3d as o3d
import torch
import math
from cvzone.PoseModule import PoseDetector

detector = PoseDetector()

# ---- 1. Inisialisasi model, mediapipe pose, dll. ----
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5
model.iou = 0.45

# Baca mesh baju
garment_mesh = o3d.io.read_triangle_mesh("Baju_2K/Baju_2K.glb")
garment_mesh.compute_vertex_normals()

# Fungsi anchor points di mesh (HARDCODE, sesuaikan index-nya!)
garment_anchors_idx = {

    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

def get_garment_anchor_points(mesh, anchor_idx_dict):
    verts = np.asarray(mesh.vertices)
    print(len(verts))
    p_left_shoulder     = verts[anchor_idx_dict["left_shoulder"]]
    p_right_shoulder    = verts[anchor_idx_dict["right_shoulder"]]
    p_left_elbow        = verts[anchor_idx_dict["left_elbow"]]
    p_right_elbow       = verts[anchor_idx_dict["right_elbow"]]
    p_left_wrist        = verts[anchor_idx_dict["left_wrist"]]
    p_right_wrist       = verts[anchor_idx_dict["right_wrist"]]
    p_left_pinky        = verts[anchor_idx_dict["left_pinky"]]
    p_right_pinky       = verts[anchor_idx_dict["right_pinky"]]
    p_left_index        = verts[anchor_idx_dict["left_index"]]
    p_right_index       = verts[anchor_idx_dict["right_index"]]
    p_left_thumb        = verts[anchor_idx_dict["left_thumb"]]
    p_right_thumb       = verts[anchor_idx_dict["right_thumb"]]
    p_left_hip          = verts[anchor_idx_dict["left_hip"]]
    p_right_hip         = verts[anchor_idx_dict["right_hip"]]
    p_left_knee         = verts[anchor_idx_dict["left_knee"]]
    p_right_knee        = verts[anchor_idx_dict["right_knee"]]
    p_left_ankle        = verts[anchor_idx_dict["left_ankle"]]
    p_right_ankle       = verts[anchor_idx_dict["right_ankle"]]
    p_left_heel         = verts[anchor_idx_dict["left_heel"]]
    p_right_heel        = verts[anchor_idx_dict["right_heel"]]
    p_left_foot_index   = verts[anchor_idx_dict["left_foot_index"]]
    p_right_foot_index  = verts[anchor_idx_dict["right_foot_index"]]
    
    anchor_points = np.vstack([
        p_left_shoulder, 
        p_right_shoulder,
        p_left_elbow,    
        p_right_elbow,
        p_left_wrist,   
        p_right_wrist,
        p_left_pinky,    
        p_right_pinky,
        p_left_index,
        p_right_index,
        p_left_thumb,    
        p_right_thumb,
        p_left_hip,      
        p_right_hip,
        p_left_knee,     
        p_right_knee,
        p_left_ankle,    
        p_right_ankle,
        p_left_heel,     
        p_right_heel,
        p_left_foot_index,
        p_right_foot_index
    ])
    return anchor_points

def best_fit_transform_with_scale(A, B):
    assert A.shape == B.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U,S,Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R)<0:
        Vt[2,:]*=-1
        R = Vt.T @ U.T
    varA = np.sum(AA**2)
    s = np.sum(S) / varA
    t = centroid_B - s*R@centroid_A
    T = np.eye(4)
    T[0:3,0:3] = s*R
    T[0:3,3] = t
    return T

# Dapatkan anchor mesh sekali di awal
mesh_anchor_pts = get_garment_anchor_points(garment_mesh, garment_anchors_idx)

# Setup Open3D Visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(width=640, height=480, visible=False)
vis.add_geometry(garment_mesh)
render_option = vis.get_render_option()
render_option.background_color = np.array([0,1,0])  # Hijau

# ---- 2. Loop webcam ----
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

posList = []

while True:
    ret, frame = cap.read()
    frame = detector.findPose(frame)
    lmList, bboxInfo = detector.findPosition(frame)
    #print(lmList)

    if bboxInfo:
        lmString = ''
        for lm in lmList:
            lmString += f'{lm[0]},{frame.shape[0] - lm[1]},{lm[2]},'
        posList.append(lmString)

    #print(len(posList))

    if not ret:
        break
    frame = cv2.flip(frame,1)
    
    # MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    
    if results.pose_world_landmarks:
        
        wls = results.pose_world_landmarks.landmark
        user_points = np.array([
            [wls[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
             wls[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
             wls[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z],

            [wls[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
             wls[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
             wls[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z],
             
            [wls[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
             wls[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
             wls[mp_pose.PoseLandmark.LEFT_ELBOW.value].z],

            [wls[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
             wls[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
             wls[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z],

            [wls[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
             wls[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
             wls[mp_pose.PoseLandmark.LEFT_WRIST.value].z],

            [wls[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
             wls[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
             wls[mp_pose.PoseLandmark.RIGHT_WRIST.value].z],

            [wls[mp_pose.PoseLandmark.LEFT_PINKY.value].x,
             wls[mp_pose.PoseLandmark.LEFT_PINKY.value].y,
             wls[mp_pose.PoseLandmark.LEFT_PINKY.value].z],

            [wls[mp_pose.PoseLandmark.RIGHT_PINKY.value].x,
             wls[mp_pose.PoseLandmark.RIGHT_PINKY.value].y,
             wls[mp_pose.PoseLandmark.RIGHT_PINKY.value].z],

            [wls[mp_pose.PoseLandmark.LEFT_INDEX.value].x,
             wls[mp_pose.PoseLandmark.LEFT_INDEX.value].y,
             wls[mp_pose.PoseLandmark.LEFT_INDEX.value].z],

            [wls[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,
             wls[mp_pose.PoseLandmark.RIGHT_INDEX.value].y,
             wls[mp_pose.PoseLandmark.RIGHT_INDEX.value].z],

            [wls[mp_pose.PoseLandmark.LEFT_THUMB.value].x,
             wls[mp_pose.PoseLandmark.LEFT_THUMB.value].y,
             wls[mp_pose.PoseLandmark.LEFT_THUMB.value].z],

            [wls[mp_pose.PoseLandmark.RIGHT_THUMB.value].x,
             wls[mp_pose.PoseLandmark.RIGHT_THUMB.value].y,
             wls[mp_pose.PoseLandmark.RIGHT_THUMB.value].z],
             

            [wls[mp_pose.PoseLandmark.LEFT_HIP.value].x,
             wls[mp_pose.PoseLandmark.LEFT_HIP.value].y,
             wls[mp_pose.PoseLandmark.LEFT_HIP.value].z],

            [wls[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
             wls[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
             wls[mp_pose.PoseLandmark.RIGHT_HIP.value].z],

            [wls[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
             wls[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
             wls[mp_pose.PoseLandmark.LEFT_KNEE.value].z],

            [wls[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
             wls[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
             wls[mp_pose.PoseLandmark.RIGHT_KNEE.value].z],

            [wls[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
             wls[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
             wls[mp_pose.PoseLandmark.LEFT_ANKLE.value].z],

            [wls[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
             wls[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
             wls[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z],

            [wls[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
             wls[mp_pose.PoseLandmark.LEFT_HEEL.value].y,
             wls[mp_pose.PoseLandmark.LEFT_HEEL.value].z],

            [wls[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
             wls[mp_pose.PoseLandmark.RIGHT_HEEL.value].y,
             wls[mp_pose.PoseLandmark.RIGHT_HEEL.value].z],

            [wls[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
             wls[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y,
             wls[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z],

            [wls[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
             wls[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y,
             wls[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z]


        ], dtype=np.float32)
       
        T = best_fit_transform_with_scale(mesh_anchor_pts, user_points)
        
      
        mesh_transformed = o3d.geometry.TriangleMesh(garment_mesh)
        mesh_transformed.transform(T)
        
        # Render di Open3D
        vis.clear_geometries()
        vis.add_geometry(mesh_transformed)
        vis.poll_events()
        vis.update_renderer()
        
        render = vis.capture_screen_float_buffer(do_render=True)
        render_np = (np.asarray(render)*255).astype(np.uint8)
        render_bgr = cv2.cvtColor(render_np, cv2.COLOR_RGB2BGR)
        
        # Overlay hijau => opsional sama seperti di kode Anda
        h1, w1 = frame.shape[:2]
        h2, w2 = render_bgr.shape[:2]
        if h2<=h1 and w2<=w1:
            roi = frame[0:h2, 0:w2]
            # Buat fungsi overlay_greenscreen() seperti di kode Anda
            blended = overlay_greenscreen(roi, render_bgr)
            frame[0:h2, 0:w2] = blended
    
    # Tampilkan ke layar
    cv2.imshow("3D Virtual Try-On", frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
vis.destroy_window()
