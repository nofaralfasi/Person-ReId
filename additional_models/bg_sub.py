from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
import glob
from finalProject.utils.preprocessing.preprocess import reduceNoise

backSub = cv2.createBackgroundSubtractorMOG2()
fgMask = None

path = "dataset/re-id/videos/ball.mp4"
capture = cv2.VideoCapture(path)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    fgMask = backSub.apply(frame)

    cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)

    extracted = None
    extracted = cv2.bitwise_and(frame, frame, extracted, mask=fgMask)
    cv2.imshow('extracted frame', extracted)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
