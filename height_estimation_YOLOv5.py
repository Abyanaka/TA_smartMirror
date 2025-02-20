import cv2
from ultralytics import YOLO

model = YOLO('yolov5s.pt')
cap = cv2.VideoCapture(1)  # Ganti dengan indeks webcam Anda

scale_factor = 0.7

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Jalankan deteksi
    results = model(frame)
    
    # Buat salinan frame untuk digambar
    annotated_frame = frame.copy()

    # Loop setiap bounding box
    for box in results[0].boxes:
        class_id = int(box.cls[0])   
        conf = float(box.conf[0])
        # 0 = 'person' di COCO
        if class_id == 0:
            x1, y1, x2, y2 = box.xyxy[0]
            width  = x2 - x1
            height = y2 - y1

            # Gambar bounding box "asli"
            cv2.rectangle(
                annotated_frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )
            
            # Hitung 'tinggi orang' berdasar scale factor
            scaled_height = height * scale_factor

            # Tulis teks: confidence, lebar, tinggi (dengan skala)
            label_text = f"Person {conf:.2f} | {int(width)}x{int(height)} -> {int(scaled_height)} scaled"
            cv2.putText(
                annotated_frame,
                label_text,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    cv2.imshow('Person Detection - Height Scale Demo', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
