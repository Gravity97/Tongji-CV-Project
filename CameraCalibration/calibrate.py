"""
this file is used to calibrate camera
"""

import glob
import cv2
import numpy as np
from CameraCalibration.chessBoard import ChessBoard


def writeIntrinsicMatrix(camera_matrix, distortion_coefficient, mode: str):
    if mode == "normal":
        with open('Result/intrinsic_normal.py', mode='w', encoding='utf-8') as f:
            f.write("import numpy as np\n")

            f.write("camera_matrix = np.float32(" + str(camera_matrix.tolist()) + ')\n')
            f.write("distortion_coefficient = np.float32(" + str(distortion_coefficient.tolist()) + ')\n')

        print("matrix has been written to intrinsic_normal.py")
    elif mode == "fisheye":
        with open('Result/intrinsic_fisheye.py', mode='w', encoding='utf-8') as f:
            f.write("import numpy as np\n")

            f.write("camera_matrix = np.float32(" + str(camera_matrix.tolist()) + ')\n')
            f.write("distortion_coefficient = np.float32(" + str(distortion_coefficient.tolist()) + ')\n')

        print("matrix has been written to intrinsic_fisheye.py")


def Calibrate(images_folder: str, cb: ChessBoard, mode="normal"):
    image_points = []  # 2D corner points
    world_points = []  # 3D world's points

    # get 3D points
    wps = np.zeros((cb.row * cb.col, 1, 3), np.float32)  # use to store 3D points coordinates for each corner
    wps[:, 0, :2] = np.mgrid[0:cb.col, 0:cb.row].T.reshape(-1, 2)
    wps = wps * cb.width  # single board size length (mm), transfer the unit from pixel to mm

    h, w = 0, 0

    image_sets = glob.glob(images_folder + '/*.png')  # images of chess boards
    for image_path in image_sets:
        image = cv2.imread(image_path)
        h, w, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # remove channels

        # append 3D points
        world_points.append(wps)

        # get corners and append
        _, corners = cv2.findChessboardCorners(gray, (cb.col, cb.row), None)  # corner: N,1,2
        image_points.append(corners)

        # error
        if len(world_points[0]) != len(image_points[0]):
            print(f"no match in {image_path}!")
            return

    if mode == "normal":
        _, camera_matrix, distortion_coefficient, rvec, tvec = cv2.calibrateCamera(
            world_points, image_points, (w, h), None, None)

        print("calibration finished!")
        print("num of boards :", len(rvec), len(tvec))
        print("K:", camera_matrix)
        print("D:", distortion_coefficient)

        writeIntrinsicMatrix(camera_matrix, distortion_coefficient, mode)
        return camera_matrix, distortion_coefficient

    elif mode == "fisheye":
        K = np.array(np.zeros((3, 3)))
        D = np.array(np.zeros((4, 1)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        rvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(len(image_sets))]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(len(image_sets))]
        ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(world_points, image_points,
                                                        (w, h), K, D, rvecs, tvecs, criteria=criteria)

        print("calibration finished!")
        print("K:", K)
        print("D:", D)

        writeIntrinsicMatrix(K, D, mode)

        return K, D


if __name__ == '__main__':
    cb = ChessBoard(9, 6, 10)  # chess board: col, row, width(mm)

    mode = "normal"
    # mode = "fisheye

    Calibrate("./Data", cb, mode)
