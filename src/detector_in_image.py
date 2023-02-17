import numpy as np
import cv2 
import os
import time

threshold = 0.5 # human face's confidence threshold

prototxt_file = os.path.join('./SSD_deploy.prototxt')
caffemodel_file = os.path.join('./model.caffemodel')
net = cv2.dnn.readNetFromCaffe(prototxt_file, caffeModel=caffemodel_file)

image = cv2.imread('/Code/Dataset/n000001/0001_01.jpg')
origin_h, origin_w = image.shape[:2]

blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

tic = time.time()
net.setInput(blob)
detections = net.forward()
print('net forward time: {:.4f}'.format(time.time() - tic))
# detection.shape = (1,1,num_bbox,7) with 7 is 2 output is face or non_face and (x,y,w,h,conf) 

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2] 
    if confidence > threshold:
        bounding_box = detections[0, 0, i, 3:7] * np.array([origin_w, origin_h, origin_w, origin_h])
        x_start, y_start, x_end, y_end = bounding_box.astype('int')
        print(x_start, y_start, x_end, y_end)

        label = '{0:.2f}%'.format(confidence * 100)
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
        cv2.rectangle(image, (x_start, y_start - 18), (x_end, y_start), (0, 0, 255), -1)
        cv2.putText(image, label, (x_start+2, y_start-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cropped_image = image[y_start:y_end, x_start:x_end]

cv2.imshow('output', cropped_image)
while True:
    if cv2.waitKey(0) & 0xFF==ord('d'):
        break
cv2.destroyAllWindows()

#cv2.imwrite('/Code/image/crop_obama2.jpg', cropped_image)