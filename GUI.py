import cv2
import numpy as np
import subprocess

# Callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is inside the button area
        if 50 < x < 200 and 50 < y < 100:
            print("Button Clicked! Running other script...")
            subprocess.run(["python", "Trial (Feb)\Body Classification.py"])  # Run the other Python file

# Initialize OpenCV window
cv2.namedWindow("What's Your Gender?")
cv2.setMouseCallback("What's Your Gender?", mouse_callback)

while True:
    # Create a blank image
    img = np.ones((400, 400, 3), dtype=np.uint8) * 200  # Gray background

    # Draw button
    cv2.rectangle(img, (50, 50), (200, 100), (0, 0, 255), -1)  # Red button
    cv2.putText(img, "Run File", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the window
    cv2.imshow("What's Your Gender?", img)

    # Close window on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
