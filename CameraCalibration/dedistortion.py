"""
this file is used to dedistortion the image
"""

import cv2
import numpy as np


def dedistortion(image, camera_mode, size):
    if camera_mode == 'normal':
        from Result.intrinsic_normal import camera_matrix, distortion_coefficient

        size = image.shape[:2]
        # we set alpha = 0 : reserve black pixels
        new_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficient,
                                                      size, 0, size,
                                                      centerPrincipalPoint=False)
        # dedistort
        mapX, mapY = cv2.initUndistortRectifyMap(camera_matrix, distortion_coefficient, None, new_matrix, size, 5)
        dedistorted_image = cv2.remap(image, mapX, mapY, cv2.INTER_LINEAR)

        print("dedistorted finished!")
        cv2.imshow("Dedistorted image", dedistorted_image)
        cv2.waitKey(0)

        return dedistorted_image

    elif camera_mode == 'fisheye':
        from Result.intrinsic_fisheye import camera_matrix, distortion_coefficient

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix, distortion_coefficient, np.eye(3), camera_matrix, size, cv2.CV_16SC2)

        dedistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        print("dedistorted finished!")
        cv2.imshow('Dedistorted Image', dedistorted_image)
        cv2.waitKey(0)

        return dedistorted_image


if __name__ == '__main__':
    image_path = ""
    camera_mode = "normal"

    image = cv2.imread(image_path)

    dedistorted_image = dedistortion(image, camera_mode, image.shape[:2])
