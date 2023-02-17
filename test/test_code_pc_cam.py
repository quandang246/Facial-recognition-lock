import numpy as np
import cv2 
import os
import time

threshold = 0.5  # human face's confidence threshold

prototxt_file = os.path.join(r"C:\Users\admin\OneDrive\Documents\Project\Face-detection-SSD-20230206T125125Z-001\Face-detection-SSD\SSD_deploy.prototxt")
caffemodel_file = os.path.join(r"C:\Users\admin\OneDrive\Documents\Project\Face-detection-SSD-20230206T125125Z-001\Face-detection-SSD\model.caffemodel")
net = cv2.dnn.readNetFromCaffe(prototxt_file, caffeModel=caffemodel_file)

url = 'rtsp://192.168.1.180:8554/live.sdp'

#url = "http://192.168.1.125:80/video"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    origin_h, origin_w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    tic = time.time()
    net.setInput(blob)
    detections = net.forward()
    print('nDet forward time: {:.4f}'.format(time.time() - tic))
    # detection.shape = (1,1,num_bbox,7) with 7 is (x,y,w,h,conf) and 2 output is face or non_face

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2] 
        if confidence > threshold:
            bounding_box = detections[0, 0, i, 3:7] * np.array([origin_w, origin_h, origin_w, origin_h])
            x_start, y_start, x_end, y_end = bounding_box.astype('int')

            label = '{0:.2f}%'.format(confidence * 100)
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)

    cv2.imshow('output', frame)
    if cv2.waitKey(1) & 0xFF==ord('a'):
        break

cap.release()
cv2.destroyAllWindows()
