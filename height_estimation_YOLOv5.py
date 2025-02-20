import cv2
from ultralytics import YOLO

model = YOLO('yolov5s.pt')
cap = cv2.VideoCapture(2)  # Ganti dengan indeks webcam Anda

scale_factor = 0.38

# Definisikan window sebelum loop
cv2.namedWindow("Person Detection - Height Scale Demo", cv2.WINDOW_NORMAL)
# Atur ukuran window, misal 1280x720
cv2.resizeWindow("Person Detection - Height Scale Demo", 720,1280)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    annotated_frame = frame.copy()

    for box in results[0].boxes:
        class_id = int(box.cls[0])   
        conf = float(box.conf[0])
        if class_id == 0:
            x1, y1, x2, y2 = box.xyxy[0]
            width  = x2 - x1
            height = y2 - y1

            cv2.rectangle(
                annotated_frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0), 2
            )
            
            scaled_height = height * scale_factor
            label_text = f"Person {conf:.2f} | {int(width)}x{int(height)} -> {int(scaled_height)} scaled"
            cv2.putText(
                annotated_frame,
                label_text,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2
            )

    # Gunakan nama window yg sama dengan namedWindow
    cv2.imshow("Person Detection - Height Scale Demo", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
