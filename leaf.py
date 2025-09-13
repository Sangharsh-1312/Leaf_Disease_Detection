#To detect disease in a leaf using color segmentation

import cv2
import numpy as np

img = cv2.imread('leaf.jpg')
resized = cv2.resize(img, dsize=(400,400))
hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

lower = np.array([10,50,50])
upper = np.array([35,235,235])
mask = cv2.inRange(hsv, lower, upper)

disease_area = cv2.bitwise_and(resized, resized, mask=mask)
contours, __ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
output = resized.copy()
cv2.drawContours(resized, contours, -1, (0,0,255), 3)

cv2.imshow('original leaf', resized)
cv2.imshow('Mask_disease_detection', mask)
cv2.imshow('detection_disease_area', disease_area)
cv2.imshow('contours', output)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Face detection

# import cv2

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# img = cv2.imread('face.jpg')
# resized_img = cv2.resize(img, dsize=(300,400))
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray,1.3,5,minSize=(30,30))


# for (x,y,w,h) in faces:
#     cv2.rectange(grey, (x,y), (x + w, y + h), (255,0,0))


# cv2.imshow('Face_detection', gray)   
# cv2.waitKey(0)
# cv2.destroyAllWindows() 

