import cv2
import numpy as np
import torch
from ultralytics import YOLO

from CalculateDistance.calculateDistance import calculate_distance
from CameraCalibration.Result.intrinsic_fisheye import camera_matrix, distortion_coefficient

# 加载模型
model = YOLO('ultralytics-main/runs/detect/train8/weights/best.pt')
model.classes = [1]

# 打开视频
cap = cv2.VideoCapture('data/video/result14.avi')

# 获取视频的宽度、高度和帧率
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 定义视频编码器和创建 VideoWriter 对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 或者使用 'MP4V'，取决于你想要的输出格式
out = cv2.VideoWriter('result_video.avi', fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 进行推理
    with torch.no_grad():
        outputs = model(frame)

    detection = outputs[0].boxes

    if detection.xyxy is not None and len(detection.xyxy) > 0:
        x, y, w, h = detection.xywh[0]
        x1, y1 = x.item(), y.item() + h.item() / 2

        # 创建一个包含单个点的浮点数组
        distorted_point = np.array([[x1, y1]], dtype=np.float32)

        # 将点的形状从 (1, 2) 改为 (1, 1, 2)
        distorted_point = distorted_point.reshape(1, 1, 2)

        # 计算去畸变后的点坐标
        undistorted_point = cv2.undistortPoints(distorted_point, camera_matrix, distortion_coefficient, P=camera_matrix)
        undistorted_point = undistorted_point.reshape(-1, 2)

        # 去畸变后的坐标
        x_undistorted, y_undistorted = undistorted_point[0]

        distance = calculate_distance(x_undistorted, y_undistorted)
        # print(distance)

        # 在frame上添加距离标签
        if 0 < distance < 10:
            frame = outputs[0].plot()
            label = f"Distance: {distance:.6f}"
            cv2.putText(frame, label, (int(x1), int(y1) + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

    out.write(frame)

# 释放资源
cap.release()
cv2.destroyAllWindows()
