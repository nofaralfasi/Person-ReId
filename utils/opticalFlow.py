import cv2
import numpy as np


def getMaskFromOpticalFlow(sequences: "array of frames sequences", isVideo: bool):
    numOfFrames = len(sequences)
    if isVideo:
        frame1 = sequences[0]
    else:
        frame1 = cv2.imread(sequences[0])

    if numOfFrames > 1:
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        for index in range(1, numOfFrames):
            if isVideo:
                frame2 = sequences[index]
            else:
                frame2 = cv2.imread(sequences[index])
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            ret, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
            # cv2.imshow('binary', binary)
            cropped = cv2.bitwise_and(frame2, frame2, mask=binary)

            cv2.imshow('cropped', cropped)
            cv2.imshow('frame2Original', frame2)

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            prvs = next
    else:
        return "Must be more then 2 frames"


def PointsOpticalFlow(sequences: "array of frames sequences", isVideo: bool):
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (50, 50)

    # fontScale
    fontScale = 1

    # Blue color in BGR
    # color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 1

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.4,
                          minDistance=7,
                          blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it

    if isVideo:
        old_frame = sequences[0]
    else:
        old_frame = cv2.imread(sequences[0])
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    numOfFrames = len(sequences)

    for index in range(1, numOfFrames):

        if isVideo:
            frame = sequences[index]
        else:
            frame = cv2.imread(sequences[index])

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            epsilon = 0.3
            a, b = new.ravel()
            c, d = old.ravel()
            if abs(c - a) > epsilon or abs(d - b) > epsilon:
                mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
                frame = cv2.putText(frame, 'id' + str(i), (a, b), font, fontScale, (255, 0, 0), thickness, cv2.LINE_AA)
        # img = cv2.add(frame, mask)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(0) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
