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


def aruco_detect(frame, gray):

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_50)
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    back = np.array([])
    
#     if ids is not None:
#         for i in range(len(ids)):
            
#             #print(corners[i])
#             #print(tz)
            
#             cv2.line(frame, (int(corners[i][0][0][0]),int(corners[i][0][0][1])), 
#                                   (int(corners[i][0][1][0]),int(corners[i][0][1][1])), (255,0,0), 2)
            
#             cv2.line(frame, (int(corners[i][0][1][0]),int(corners[i][0][1][1])), 
#                                   (int(corners[i][0][2][0]),int(corners[i][0][2][1])), (0,255,0), 2)

#             cv2.line(frame, (int(corners[i][0][2][0]),int(corners[i][0][2][1])), 
#                                   (int(corners[i][0][3][0]),int(corners[i][0][3][1])), (0,0,255), 2)

#             cv2.putText(frame, str(ids[i]), (int(corners[i][0][0][0]),int(corners[i][0][0][1])), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

#             x = corners[i][0][0][0] - corners[i][0][1][0]
#             y = corners[i][0][0][1] - corners[i][0][1][1]
#             dis = math.sqrt(pow(x,2)+pow(y,2))*100 + 3
            
#             # modified distance
#             dis = 0.0084*pow(dis, 2) + 0.5877*dis - 2.1246
# #                 print('id = ', ids[i], dis, '(cm)')
#             id = int(ids[i])
#             #if (0 <= id < 22) & (tz < 90):
#             back = np.array([id, dis])
# #                 distance[i, :] = np.array([id, dis, tz])
# #         distance = np.round(distance, 1)
    if ids is not None:
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.03, mtx, dist)
            for i in range(len(ids)):
                x = tvec[i, 0, 0]
                y = tvec[i, 0, 1]
                z = tvec[i, 0, 2]
                R = cv2.Rodrigues(rvec[i])[0]
                # euler angle
#                 tx = atan2(R[2, 1], R[2, 2])
#                 ty = atan2(-R[2, 0], sqrt(pow(R[2, 1], 2) + pow(R[2, 2], 2)))
                tz = np.rad2deg(math.atan2(R[1, 0], R[0, 0]))
#                 angle = np.rad2deg(np.array([tx, ty, tz]))
#                 print(tz)
                dis = math.sqrt(pow(x,2)+pow(y,2)+pow(z,2))*100 + 3
                # modified distance
                dis = 0.0084*pow(dis, 2) + 0.5877*dis - 2.1246
#                 print('id = ', ids[i], dis, '(cm)')
                id = int(ids[i])
                if (0 <= id < 22) & (tz < 90):
                    back = np.array([id, dis])
#                 distance[i, :] = np.array([id, dis, tz])
#         distance = np.round(distance, 1)
    return frame, back
    
while 1:

    ret, frame = cap.read()
    
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame, back = aruco_detect(frame, gray)

        cv2.imshow('img',frame)

        cv2.waitKey(1)
        
        if len(back):
            print("id: %s, dis: %s"%(back[0],back[1]))
    
