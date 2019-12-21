""" **Matchers**
#Kaze Matcher for binary classification - orb , kaze ,brief,fast
"""
import cv2
import numpy as np

from finalProject.classes.enumTypeKeyPoints import NamesAlgorithms
from finalProject.utils.keyPoints.AlgoritamKeyPoints import SurfDetectKeyPoints
from finalProject.utils.keyPoints.AlgoritamKeyPoints import SiftDetectKeyPoints
from finalProject.utils.keyPoints.AlgoritamKeyPoints import ORBDetectKeyPoints
from finalProject.utils.keyPoints.AlgoritamKeyPoints import KazeDetectKeyPoints
from finalProject.utils.keyPoints.AlgoritamKeyPoints import SurfDetectKeyPoints


def KazeMatcher(desc1, desc2):
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    nn_matches = matcher.knnMatch(desc1, desc2, 2)
    return nn_matches


"""# BF MATCHER"""


def findClosesHuman(human, myPeople, config: "config file"):
    keyTarget, descriptionTarget = SurfDetectKeyPoints(human["frame"])
    # TODO check with Liran if it's a different  DescriptionTarget here or not
    if keyTarget is None or descriptionTarget is None:
        return None  # dont have key points for this human
    maxMatch = []
    for p in myPeople:
        # remove trace frames
        if len(p.frames) > config["max_length_frames"]:
            p.history.extend(p.frames[0:len(p.frames) - config["max_length_frames"]])
            p.frames = p.frames[-config["max_length_frames"]:]

        MatchP = []
        for frame in p.frames:
            kp, dp = SurfDetectKeyPoints(frame)
            if kp is None or dp is None:
                continue
            else:
                goodMatch = FLANNMATCHER(descriptionTarget, dp, config["FlannMatcherThreshold"])
            if len(keyTarget) == 0:
                acc = 0
            else:
                acc = len(goodMatch) / len(keyTarget)
            MatchP.append(acc)
        if len(MatchP) > 0:
            MeanAcc = np.mean(MatchP)
        else:
            MeanAcc = 0
        maxMatch.append((p, MeanAcc))

    return maxMatch


def BFMatcher(des1, des2, threshold):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append([m])
    return good


"""#FLANN MATCHER for SURF and SIFT"""


def FLANNMATCHER(des1, des2, threshold=0.8):  # threshold is the distance between the points we're comparing
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    if len(des1) >= 2 and len(des2) >= 2:
        matches = flann.knnMatch(des1, des2, k=2)
        # Need to draw only good matches, so create a mask
        good = []
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < threshold * n.distance:
                good.append([m])
        return good
    else:
        return []


def compareBetweenTwoFramesObject(sourceFrame, targetFrame):
    binaryAlgo = [NamesAlgorithms.ORB.name, NamesAlgorithms.KAZE.name]
    floatAlgo = [NamesAlgorithms.SURF.name, NamesAlgorithms.SIFT.name]
    results = []
    for algo in binaryAlgo:
        des_s = sourceFrame[algo]["des"]
        des_t = targetFrame[algo]["des"]
        if len(des_s) == 0 or len(des_t) == 0:
            results.append(0)
        else:
            matches = KazeMatcher(des_s, des_t)
            acc = len(matches) / len(des_t)
            results.append(acc)

    for algo in floatAlgo:
        des_s = sourceFrame[algo]["des"]
        des_t = targetFrame[algo]["des"]
        if len(des_s) == 0 or len(des_t) == 0:
            results.append(0)
        else:
            matches = FLANNMATCHER(des_s, des_t)
            acc = len(matches) / len(des_t)
            results.append(acc)

    return np.mean(results)


def CompareBetweenTwoDescription(sourceDescriptor, targetDescriptor):
    acc_target = {}
    for _id, target in targetDescriptor.items():
        tableAcc = np.zeros(shape=[len(target), len(sourceDescriptor[0])])
        for index_t, frame_t in enumerate(target):
            for index_s, frame_s in enumerate(sourceDescriptor[0]):
                tableAcc[index_t, index_s] = compareBetweenTwoFramesObject(frame_s, frame_t)

        maxAcc = np.amax(tableAcc)
        ind = np.unravel_index(np.argmax(maxAcc, axis=None), maxAcc.shape)
        acc_target[_id] = {"maxAcc ": maxAcc, "target": target, "indexMax": ind}
    return acc_target
