import cv2
import numpy as np
import textwrap
import mediapipe as mp

# === Fungsi Pendukung untuk Tampilan GUI ===
def roundedRect(img, top_left, bottom_right, color, radius):
    """
    Menggambar kotak penuh dengan sudut membulat pada image.
    img         : Citra (image) tempat menggambar.
    top_left    : Koordinat (x, y) pojok kiri atas.
    bottom_right: Koordinat (x, y) pojok kanan bawah.
    color       : Warna kotak (B, G, R).
    radius      : Jari-jari sudut membulat.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Gambar bagian tengah (rectangle) yang tidak membutuhkan pembulatan
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness=cv2.FILLED)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness=cv2.FILLED)
    
    # Gambar empat lengkungan di setiap sudut dengan circle
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness=cv2.FILLED)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness=cv2.FILLED)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness=cv2.FILLED)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness=cv2.FILLED)

def textWrap(frame, text, x, y, max_chars_per_line,
             font=cv2.FONT_HERSHEY_SIMPLEX, 
             font_scale=0.3,
             thickness=1, 
             color_text=(255, 255, 255), 
             color_bg=(0, 0, 0),
             padding=10, 
             alpha=0.5, 
             max_lines=5, 
             radius=20):
    """
    Menggambar teks yang di-wrap (pembungkus) menjadi beberapa baris dengan
    background semi transparan yang memiliki sudut membulat.
    
    Parameter:
    - frame: Citra input.
    - text : Teks yang akan ditampilkan.
    - (x, y): Titik awal (koordinat) teks (bagian atas dari garis dasar teks).
    - max_chars_per_line: Batas maksimal karakter per baris.
    - max_lines: Maksimal jumlah baris yang ditampilkan.
    - padding: Margin tambahan di sekitar teks.
    - alpha  : Faktor transparansi (0.0 transparan, 1.0 opaque).
    - radius : Jari-jari untuk sudut membulat.
    """
    # Bungkus teks menjadi list baris menggunakan modul textwrap
    wrapped_lines = textwrap.wrap(text, width=max_chars_per_line)
    if len(wrapped_lines) > max_lines:
        wrapped_lines = wrapped_lines[:max_lines]
    
    # Hitung ukuran tiap baris teks dan tentukan lebar maksimum
    line_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in wrapped_lines]
    text_width = max(size[0] for size in line_sizes)
    line_height = line_sizes[0][1]  # Asumsikan tinggi per baris seragam
    
    # Tentukan koordinat background dengan margin padding
    rect_x1 = x - padding
    rect_y1 = y - line_height - padding
    rect_x2 = x + text_width + padding
    rect_y2 = y + len(wrapped_lines) * line_height + padding

    # Buat overlay salinan dari frame
    overlay = frame.copy()

    # Gambar kotak dengan rounded edges di overlay
    roundedRect(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), color_bg, radius)
    
    # Gabungkan overlay ke frame asli untuk efek semi transparan
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Gambar setiap baris teks secara terpisah dengan offset vertikal
    for i, line in enumerate(wrapped_lines):
        line_y = y + i * (line_height + 5)
        cv2.putText(frame, line, (x, line_y), font, font_scale, color_text, thickness)

# === Setup Virtual Try-On 2D Menggunakan MediaPipe Pose ===
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

# Buka kamera (pastikan device index sesuai)
cap = cv2.VideoCapture(2)

# Variabel konfigurasi untuk garment
garment_scale = 1.0
offset_y = -25  

# Inisialisasi window full screen
windowName = "2D Virtual Try-On with GUI"
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame secara horizontal agar interaksi lebih natural
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Ubah frame ke RGB untuk proses MediaPipe Pose
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Gambar pose landmarks pada frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

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

        # Ambil titik tumit kiri dan kanan
        left_heel = np.array([
            lm[mp_pose.PoseLandmark.LEFT_HEEL.value].x * w,
            lm[mp_pose.PoseLandmark.LEFT_HEEL.value].y * h
        ], dtype=np.float32)
        right_heel = np.array([
            lm[mp_pose.PoseLandmark.RIGHT_HEEL.value].x * w,
            lm[mp_pose.PoseLandmark.RIGHT_HEEL.value].y * h
        ], dtype=np.float32)
        bottom_center = (left_heel + right_heel) / 2

        # Tentukan titik tujuan untuk overlay garment
        pts_dst = np.float32([
            left_shoulder + np.array([0, offset_y]),
            right_shoulder + np.array([0, offset_y]),
            bottom_center
        ])

        center_shoulder = (left_shoulder + right_shoulder) / 2
        pts_dst_scaled = center_shoulder + garment_scale * (pts_dst - center_shoulder)
        pts_dst_scaled = np.float32(pts_dst_scaled)

        # Titik referensi garment (disesuaikan dengan struktur garment)
        garment_h, garment_w = garment.shape[:2]
        pts_src = np.float32([
            [garment_w * 0.3, 0],              # Titik atas kiri garment (wakili bahu kiri)
            [garment_w * 0.7, 0],              # Titik atas kanan garment (wakili bahu kanan)
            [garment_w * 0.5, garment_h]       # Titik bawah garment (bagian bawah dress)
        ])

        # Hitung matriks transformasi afine
        M = cv2.getAffineTransform(pts_src, pts_dst_scaled)
        
        # Warp garment ke ukuran frame sesuai titik-titik tujuan
        warped_garment = cv2.warpAffine(garment, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        
        # Overlay garment ke frame menggunakan alpha channel
        if warped_garment.shape[2] == 4:
            garment_rgb = warped_garment[:, :, :3]
            alpha_mask = warped_garment[:, :, 3] / 255.0
            for c in range(3):
                frame[:, :, c] = (alpha_mask * garment_rgb[:, :, c] +
                                  (1 - alpha_mask) * frame[:, :, c])
        else:
            frame = cv2.addWeighted(frame, 1, warped_garment, 0.5, 0)

    # === Tambahkan Teks dengan Background Rounded (GUI) ===
    caption = ("Untuk tipe tubuh ideal, dapat menggunakan pakaian dengan ukuran yang pas (well-fitted) "
               "dengan tubuh. Tidak terlalu besar, maupun terlalu kecil")
    # Posisi dan batas karakter per baris untuk teks
    x, y = 5, 400
    max_chars_per_line = 50
    textWrap(frame, caption, x, y, max_chars_per_line)

    cv2.imshow(windowName, frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
