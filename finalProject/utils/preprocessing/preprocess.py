import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def framesExists(frames):
    return len(frames) > 0


def readFromInputVideoFrames(config):
    frames = []

    if config["isVideo"]:
        cap = cv2.VideoCapture(config["inputVideo"])
        i = 0
        while i < config["skipRateFrameFromBeginning"]:
            ret, frame = cap.read()
            i += 1

        i = 0
        while ret and i < config["videoFrameLength"]:
            frames.append(frame)
            ret, frame = cap.read()
            i += 1
    else:
        path = config["inputVideo"]
        for (dirpath, dirnames, filenames) in os.walk(path):
            frames.extend(filenames)
            break
        frames.sort()
        frames = list(map(lambda file: path + "/" + file, frames))

    return frames


def reduceNoise(frames):
    framesMask = []
    backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=100)

    for f in frames:
        extracted = None
        fgMask = backSub.apply(f)
        extracted = cv2.bitwise_and(f, f, extracted, mask=fgMask)
        framesMask.append(extracted)

    return framesMask
