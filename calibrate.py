import cv2
import numpy as np
import glob
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"


square_count_x = 10
square_count_y = 7

nx = square_count_x - 1
ny = square_count_y - 1

square_size = 0.025
criteria = (cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

object_points_3d = np.zeros((nx * ny, 3), np.float32)
object_points_3d[:, :2] = np.mgrid[0:ny, 0:nx].T.reshape(-1, 2)
object_points_3d = object_points_3d * square_size

object_points = []
image_points = []


def main():
    images = glob.glob("./Kalibrasi Kamera/*.jpg")
    for img in images:
        image = cv2.imread(img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        success, corners = cv2.findChessboardCorners(gray, (ny, nx), None)
        if success:
            object_points.append(object_points_3d)
            corners_2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            image_points.append(corners_2)
            cv2.drawChessboardCorners(image, (ny, nx), corners_2, success)
            cv2.imshow("Image", image)
            cv2.waitKey(1000)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, gray.shape[::-1], None, None
    )

    cv_file = cv2.FileStorage("calibration_chessboard.yaml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    cv_file.release()

    cv_file = cv2.FileStorage("calibration_chessboard.yaml", cv2.FILE_STORAGE_READ)
    mtx = cv_file.getNode("K").mat()
    dist = cv_file.getNode("D").mat()
    cv_file.release()

    print("Camera matrix:")
    print(mtx)

    print("\n Distortion coefficient:")
    print(dist)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    print(__doc__)
    main()