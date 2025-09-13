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
