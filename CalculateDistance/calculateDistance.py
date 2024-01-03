"""
this file is used to calculate the distance
"""

import cv2
import numpy as np
from SolveHomography.result import homography_matrix


def calculate_distance(x, y):
    homogeneous_coordinate = np.array([x, y, 1])

    world_point = np.dot(homography_matrix, homogeneous_coordinate)

    ratio = 1 / world_point[2]
    world_point *= ratio

    # return np.sqrt(world_point[0] ** 2 + world_point[1] ** 2) / 1000  # transfer mm into m
    return world_point[1] / 1000  # transfer mm into m

