""" **Matchers**

#Kaze Matcher for binary classification - orb , kaze ,brief,fast
"""
import cv2
import numpy as np
from finalProject.utils.keyPoints.AlgoritamKeyPoints import SurfDetectKeyPoints


def KazeMatcher(desc1, desc2):
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    nn_matches = matcher.knnMatch(desc1, desc2, 2)
    return nn_matches


"""# BF MATCHER"""


def findClosesHuman(human, myPeople, config: "config file"):
    keyTarget, DescriptionTarget = SurfDetectKeyPoints(human["frame"])
    if keyTarget is None or DescriptionTarget is None:
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
                goodMatch = FLANNMATCHER(DescriptionTarget, dp, config["FlannMatcherThreshold"])
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


def FLANNMATCHER(des1, des2, threshold):  # threshold is the distance between the points we're comparing
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
