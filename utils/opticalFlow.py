import cv2
import numpy as np


def getMaskFromOpticalFlow(sequences: "array of frames sequences"):
    numOfFrames = len(sequences)
    frame1 = cv2.imread(sequences[0])
    if numOfFrames > 1:
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        for index in range(1, numOfFrames):
            frame2 = cv2.imread(sequences[index])
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
            cv2.imshow('binary', binary)
            cropped = cv2.bitwise_and(frame2,frame2,mask=binary)

            cv2.imshow('cropped', cropped)
            cv2.imshow('frame2Original', frame2)

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            prvs = next
    else:
        return "Must be more then 2 frames"
