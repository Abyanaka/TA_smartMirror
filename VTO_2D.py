import cv2
import mediapipe as mp
import numpy as np

# ----- Inisialisasi -----
# Baca gambar garment (dress) dengan channel alpha (pastikan file memiliki transparansi)
garment = cv2.imread("Baju 2D/DressYellow.png", cv2.IMREAD_UNCHANGED)
if garment is None:
    raise FileNotFoundError("Garment image not found. Check the path.")

# Inisialisasi MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(2)

garment_scale = 1.0
offset_y = -25  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame secara horizontal agar interaksi lebih natural
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Konversi frame ke RGB lalu proses dengan MediaPipe Pose
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # (Opsional) Gambar landmark pada frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # ----- Ekstrak landmark yang diperlukan -----
        lm = results.pose_landmarks.landmark

        # Ambil titik bahu kiri dan kanan
        left_shoulder = np.array([
            lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
            lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h
        ], dtype=np.float32)
        right_shoulder = np.array([
            lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
            lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h
        ], dtype=np.float32)

        # Ambil titik heel (kaki) sebagai titik bawah garment
        left_heel = np.array([
            lm[mp_pose.PoseLandmark.LEFT_HEEL.value].x * w,
            lm[mp_pose.PoseLandmark.LEFT_HEEL.value].y * h
        ], dtype=np.float32)
        right_heel = np.array([
            lm[mp_pose.PoseLandmark.RIGHT_HEEL.value].x * w,
            lm[mp_pose.PoseLandmark.RIGHT_HEEL.value].y * h
        ], dtype=np.float32)
        bottom_center = (left_heel + right_heel) / 2

        # ----- Tentukan titik tujuan (destination points) -----
        # Tambahkan offset ke titik bahu agar garment naik (offset_y negatif untuk naik)
        pts_dst = np.float32([
            left_shoulder + np.array([0, offset_y]),
            right_shoulder + np.array([0, offset_y]),
            bottom_center
        ])

        # Jika ingin menggunakan scaling tambahan (di sini tidak ada perubahan, scale 1.0)
        center_shoulder = (left_shoulder + right_shoulder) / 2
        pts_dst_scaled = center_shoulder + garment_scale * (pts_dst - center_shoulder)
        pts_dst_scaled = np.float32(pts_dst_scaled)

        # ----- Definisikan titik jangkar garment (source points) -----
        # Misal asumsikan bagian atas garment (garment image) mewakili bahu di bagian atas gambar
        garment_h, garment_w = garment.shape[:2]
        pts_src = np.float32([
            [garment_w * 0.3, 0],              # Titik atas kiri garment (wakili bahu kiri)
            [garment_w * 0.7, 0],              # Titik atas kanan garment (wakili bahu kanan)
            [garment_w * 0.5, garment_h]       # Titik bawah garment (bagian bawah dress)
        ])

        # ----- Hitung matriks transformasi afine -----
        M = cv2.getAffineTransform(pts_src, pts_dst_scaled)
        
        # Warp garment ke ukuran frame sesuai titik-titik tujuan
        warped_garment = cv2.warpAffine(garment, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        
        # ----- Overlay garment ke frame menggunakan alpha channel -----
        if warped_garment.shape[2] == 4:
            garment_rgb = warped_garment[:, :, :3]
            alpha_mask = warped_garment[:, :, 3] / 255.0
            for c in range(3):
                frame[:, :, c] = (alpha_mask * garment_rgb[:, :, c] +
                                  (1 - alpha_mask) * frame[:, :, c])
        else:
            frame = cv2.addWeighted(frame, 1, warped_garment, 0.5, 0)

    cv2.imshow("2D Virtual Try-On", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
