"""
this file is used to calculate the distance
"""

import cv2
import numpy as np
from SolveHomography.result import homography_matrix


def calculate_distance(image_point: list):
    homogeneous_coordinate = np.array([image_point, 1])

    world_point = np.dot(homography_matrix, homogeneous_coordinate)

    ratio = 1 / world_point[2]
    world_point *= ratio

    return np.sqrt(world_point[0] ** 2 + world_point[1] ** 2) / 1000  # transfer mm into m


