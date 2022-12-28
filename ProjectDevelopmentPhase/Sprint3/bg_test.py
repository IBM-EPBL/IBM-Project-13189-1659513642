from __future__ import print_function
import cv2
import argparse
from skimage.transform import rotate

backSub = cv2.createBackgroundSubtractorKNN()
# backSub = cv.createBackgroundSubtractorMOG2()
capture = cv2.VideoCapture(0)
cnt=1
if not capture.isOpened():
    exit(0)
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    fgMask = backSub.apply(frame)
    
    cv2.imshow('Frame', frame)
    cv2.imshow('Hand Segmentation', fgMask)
    # cv2.imshow('KNN', fgMask)
    cv2.imwrite("Application\\temp_dataset2\\"+str(cnt)+".png", cv2.rotate(fgMask, cv2.ROTATE_180))
    cnt+=1
    
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
