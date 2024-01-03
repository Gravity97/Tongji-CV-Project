"""
this is file is to solve homography matrix, you need to set the corresponding points in the code
"""

import cv2
import numpy as np

is_check = False  # if you want to check the points you have set, set it True


def solve_homography():
    imagePoints = [(397, 538), (292, 521), (308, 433), (379, 440), (372, 391), (458, 449),
                   (512, 555)]  # 2d points in image plane.
    worldPoints = [(0, 600), (-300, 600), (-300, 900), (0, 900), (0, 1200), (300, 900),
                   (300, 600)]  # 3d points in real world space(ground plane z=0)

    imagePoints = np.array(imagePoints, dtype=np.float32)  # change type to np.ndarray
    worldPoints = np.array(worldPoints, dtype=np.float32)

    H, _ = cv2.findHomography(imagePoints, worldPoints)
    cv2.waitKey(0)

    with open('result.py', mode='w', encoding='utf-8') as f:
        f.write("import numpy as np\n")

        f.write("homography_matrix = np.float32(" + str(H.tolist()) + ')')

    print("homography matrix has been written to result.py")


def click_corner(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(image, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(image, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)


if __name__ == '__main__':
    file = "Data/8_dedistorted.png"  # the image to use

    if not is_check:
        solve_homography()
    else:
        # check the points
        image = cv2.imread(file)
        cv2.destroyAllWindows()

        cv2.namedWindow("solveHomography")
        cv2.setMouseCallback("solveHomography", click_corner)

        while True:
            cv2.imshow("solveHomography", image)
            key = cv2.waitKey(1) & 0xff
            if key == ord('q') or key == ord('Q'):
                break
