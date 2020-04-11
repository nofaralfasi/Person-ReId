import os
import cv2
import numpy as np


def check_frames_exist(frames):
    return len(frames) > 0


def read_frames_from_video(config):
    frames = []

    if config["isVideo"]:
        cap = cv2.VideoCapture(config["inputVideo"])
        i = 0
        while i < config["skipRateFrameFromBeginning"]:
            ret, frame = cap.read()  # cap.read() returns a bool. If read correctly - True. You can check end of the video by it.
            i += 1

        i = 0
        while ret and i < config["videoFrameLength"]:
            frames.append(frame)
            ret, frame = cap.read()  # grabs, decodes and returns the next video frame.
            i += 1

    else:
        path = config["inputVideo"]
        for (dirpath, dirnames, filenames) in os.walk(path):
            frames.extend(filenames)
            break

        frames.sort()
        frames = list(map(lambda file: path + "/" + file, frames))

    return frames


def reduce_noise(frames):
    framesMask = []
    backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=100)

    for f in frames:
        extracted = None
        fgMask = backSub.apply(f)
        extracted = cv2.bitwise_and(f, f, extracted, mask=fgMask)
        framesMask.append(extracted)

    return framesMask


def removeRemovalColor(frames):
    newFrames = []
    sensitivity = 15
    lower_green = np.array([60 - sensitivity, 100, 50])
    upper_green = np.array([60 + sensitivity, 255, 255])

    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.bitwise_not(mask,mask)
        blacksIndex = np.where(mask == 0)
        frame[blacksIndex] = 0
        newFrames.append(frame)

    return newFrames
