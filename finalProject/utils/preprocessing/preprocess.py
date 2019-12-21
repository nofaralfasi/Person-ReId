import cv2
import os


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
        while ret:
            frames.append(frame)
            ret, frame = cap.read()
    else:
        path = config["inputVideo"]
        for (dirpath, dirnames, filenames) in os.walk(path):
            frames.extend(filenames)
            break
        frames.sort()
        frames = list(map(lambda file: path + "/" + file, frames))

    return frames
