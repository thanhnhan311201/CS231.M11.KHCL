import cv2
import numpy as np

img = cv2.imread('img.png')
img1 = img.copy()
img = -img
img1 = 255 - img1
# img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imshow('img', img)
cv2.imshow('img1', img1)
# cv2.imshow('ho_image', img[::-1,:])
# cv2.imshow('ver_image', img[,])
cv2.waitKey(0)
cv2.destroyAllWindows()