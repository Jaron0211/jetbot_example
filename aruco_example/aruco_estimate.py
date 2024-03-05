import cv2
import cv2.aruco as aruco
import numpy as np
import math
file = np.load('calibration.npz')
mtx, dist, R, T = [file[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

def rotationVectorToEulerAngles(rvec):
    R = np.zeros((3, 3), dtype=np.float64)
    cv2.Rodrigues(rvec, R)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:  # 偏航，俯仰，滚动
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    # 偏航，俯仰，滚动换成角度
    rx = x * 180.0 / 3.141592653589793
    ry = y * 180.0 / 3.141592653589793
    rz = z * 180.0 / 3.141592653589793
    return rx, ry, rz

def aruco_detect(frame, gray):

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_50)
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    back = np.array([])

    if ids is not None:
                    
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.035, mtx, dist)

        for i in range(rvec.shape[0]):
            aruco.drawAxis(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.03)
            aruco.drawDetectedMarkers(frame, corners)

        ###### DRAW ID #####
        cv2.putText(frame, "Id: " + str(ids), (0, 40), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        EulerAngles = rotationVectorToEulerAngles(rvec)
        EulerAngles = [round(i, 2) for i in EulerAngles]
        cv2.putText(frame, "Attitude_angle:" + str(EulerAngles), (0, 120), font, 0.6, (0, 255, 0), 2,
                    cv2.LINE_AA)
        tvec = tvec * 1000

        for i in range(3):
            tvec[0][0][i] = round(tvec[0][0][i], 1)
        tvec = np.squeeze(tvec)
        cv2.putText(frame, "Position_coordinates:" + str(tvec) + str('mm'), (0, 80), font, 0.6, (0, 255, 0), 2,
                    cv2.LINE_AA)
    return frame
    
while 1:

    ret, frame = cap.read()
    
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame, back = aruco_detect(frame, gray)

        cv2.imshow('img',frame)

        cv2.waitKey(1)
    
