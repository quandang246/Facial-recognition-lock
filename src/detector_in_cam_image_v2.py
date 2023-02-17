import numpy as np
import cv2 
import os
import time

threshold = 0.5

prototxt_file = os.path.join('./Resnet_SSD_deploy.prototxt')
caffemodel_file = os.path.join('./Res10_300x300_SSD_iter_140000.caffemodel')
net = cv2.dnn.readNetFromCaffe(prototxt_file, caffeModel=caffemodel_file)

capture = cv2.VideoCapture(0)

num = 0
while True: 
    ret, frame = capture.read()
    
    origin_h, origin_w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()
    # detection.shape = (1,1,num_bbox,7) with 7 is (x,y,w,h,conf) and 2 output is face or non_face
    
    if np.max(detections[0,0,:,2]) > 0.5:
        for i in range(1):
            frame = cv2.flip(frame, 1)
            cv2.imwrite('./storage/face.png', frame)
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2] 
            if confidence > threshold:
                bounding_box = detections[0, 0, i, 3:7] * np.array([origin_w, origin_h, origin_w, origin_h])
                x_start, y_start, x_end, y_end = bounding_box.astype('int')

                label = '{0:.2f}%'.format(confidence * 100)
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
                cv2.rectangle(frame, (x_start, y_start - 18), (x_end, y_start), (0, 0, 255), -1)
                cv2.putText(frame, label, (x_start+2, y_start-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.imshow('output', frame)
        if cv2.waitKey(3500) & 0xFF==ord('d'):
            break
        cv2.destroyAllWindows()
        os.remove('./storage/face.png')
                
    else:
        print('non_face')
    
    time.sleep(5)
    
    

    

 