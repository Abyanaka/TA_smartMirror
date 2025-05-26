import cv2
import mediapipe as mp
import numpy as np
import subprocess
import sys

camera_matrix = np.array([
    [1.37048527e+03, 0.00000000e+00, 9.61743100e+02],
    [0.00000000e+00, 1.37063999e+03, 5.43861352e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])
dist_coeffs = np.array([[ 1.57437343e-01, -8.92899556e-01, -4.33763266e-03,
                         -5.90548476e-04,  1.02831667e+00 ]])


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


SCALE_FACTOR_HEIGHT = 0.45
SCALE_FACTOR_WAIST  = 0.36
SCALE_FACTOR_SHOULDER = 0.38
WAIST_INTERP_FACTOR  = 0.451

def calculate_wsr(shoulder_cm, waist_cm):
    return waist_cm / shoulder_cm 

def classify_body_type(wsr):
    if wsr < 0.62:
        return "Not-Ideal"
    elif 0.62 <= wsr <= 0.73:
        return "Ideal"
    else:
        return "Not-Ideal"

def stabilize_classification(wsr_values):
    avg_wsr = np.mean(wsr_values)
    body_type = classify_body_type(avg_wsr)
    return body_type


def launch_vto(gender, body_type):
    mapping = {
        ('woman','Underweight'): 'VTO_underWoman',
        ('woman','Ideal'):       'VTO_idealWoman',
        ('woman','Overweight'):  'VTO_overWoman',
        ('man','Underweight'):   'VTO_underMan',
        ('man','Ideal'):         'VTO_idealMan',
        ('man','Overweight'):    'VTO_overMan',
    }
    script = mapping.get((gender, body_type))
    if script:
        subprocess.run([sys.executable, f"{script}.py"])
    else:
        print(f"No VTO script for ({gender}, {body_type})")


def main():
    cap = cv2.VideoCapture(0)
    gender = None


    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, "Press 'f' = Woman",
                    (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
        cv2.putText(frame, "Press 'm' = Man", 
                    (20,80),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.putText(frame, "Press 'q' = Quit", 
                    (20,110),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.namedWindow("Select Gender", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select Gender", 1080, 1920)
        cv2.imshow("Select Gender", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('f'):
            gender = "woman"; break
        if key == ord('m'):
            gender = "man";   break
        if key == ord('q'):
            cap.release(); cv2.destroyAllWindows(); return


    cv2.namedWindow("Classification", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Classification", 1080, 1920)

    wsr_values = []
    selected_type = None
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        frame = cv2.flip(frame, 1)
        

        und = cv2.undistort(frame, camera_matrix, dist_coeffs)
        rgb = cv2.cvtColor(und, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = results.pose_landmarks.landmark

            lhx = lm[mp_pose.PoseLandmark.LEFT_HIP].x * w
            rhx = lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w
            lsx = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w
            rsx = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w
            lw = lhx + WAIST_INTERP_FACTOR*(lsx - lhx)
            rw = rhx + WAIST_INTERP_FACTOR*(rsx - rhx)
            waist_px = abs(lw - rw)

            shoulder_px = abs(lsx - rsx)

            nose_y = lm[mp_pose.PoseLandmark.NOSE].y * h
            heel_y = 0.5*(lm[mp_pose.PoseLandmark.LEFT_HEEL].y + lm[mp_pose.PoseLandmark.RIGHT_HEEL].y)*h
            height_px = abs(nose_y - heel_y)

            shoulder_cm = np.pi * shoulder_px * SCALE_FACTOR_SHOULDER
            waist_cm  = np.pi * waist_px  * SCALE_FACTOR_WAIST
            height_cm = height_px * SCALE_FACTOR_HEIGHT

            wsr = calculate_wsr(shoulder_cm, waist_cm)
            wsr_values.append(wsr)
            if len(wsr_values) > 10: wsr_values.pop(0)

            print(f"Shoulder: {shoulder_cm:.0f} cm, Height: {height_cm:.0f} cm, Waist: {waist_cm:.0f}")

            body_type = stabilize_classification(wsr_values)
            cv2.putText(img, f"Type: {body_type}", (20,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            cv2.putText(img, f"wsr: {wsr:.2f}", (20,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            cv2.putText(img, "Press 's' to Save & Launch VTO", (20,110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
        cv2.imshow("Classification", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            selected_type = body_type
            break
        if key == ord('q'):
            cap.release(); cv2.destroyAllWindows(); return

    cap.release()
    cv2.destroyAllWindows()

    launch_vto(gender, selected_type)


if __name__ == "__main__":
    main()
