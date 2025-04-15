import cv2
import numpy as np
import textwrap

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
    total_text_height = line_height * len(wrapped_lines)
    
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

# Contoh penggunaan: menampilkan teks berbaris di webcam
cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Contoh caption yang panjang dan perlu di-wrap
    caption = ("Untuk tipe tubuh ideal, dapat menggunakan pakaian dengan ukuran yang pas (well-fitted) "
    "dengan tubuh. Tidak terlalu besar, maupun terlalu kecil")
    
    # Tentukan posisi teks dan batas karakter per baris
    x, y = 5, 400
    max_chars_per_line = 50

    # Gambar teks beserta background dengan rounded edges
    textWrap(frame, caption, x, y, max_chars_per_line)

    cv2.imshow("Rounded Wrapped Caption", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
