import cv2
import numpy as np
import torch
from ultralytics import YOLO

from CalculateDistance.calculateDistance import calculate_distance
from CameraCalibration.Result.intrinsic_fisheye import camera_matrix, distortion_coefficient

# load model
model = YOLO('ultralytics-main/runs/detect/train8/weights/best.pt')
model.classes = [1]

# open video
cap = cv2.VideoCapture('data/video/result14.avi')

# get video information
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('result_video.avi', fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # detect
    with torch.no_grad():
        outputs = model(frame)

    detection = outputs[0].boxes

    if detection.xyxy is not None and len(detection.xyxy) > 0:
        x, y, w, h = detection.xywh[0]
        x1, y1 = x.item(), y.item() + h.item() / 2

        # create distorted point
        distorted_point = np.array([[x1, y1]], dtype=np.float32)

        # transfer distorted point into homogeneous coordinate
        distorted_point = distorted_point.reshape(1, 1, 2)

        # calculate undistorted point
        undistorted_point = cv2.undistortPoints(distorted_point, camera_matrix, distortion_coefficient, P=camera_matrix)
        undistorted_point = undistorted_point.reshape(-1, 2)

        # get undistorted point
        x_undistorted, y_undistorted = undistorted_point[0]

        # calculate distance
        distance = calculate_distance(x_undistorted, y_undistorted)
        # print(distance)

        # add label on the frame
        if 0 < distance < 10:
            frame = outputs[0].plot()
            label = f"Distance: {distance:.6f}"
            cv2.putText(frame, label, (int(x1), int(y1) + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

    out.write(frame)

# 释放资源
cap.release()
cv2.destroyAllWindows()
