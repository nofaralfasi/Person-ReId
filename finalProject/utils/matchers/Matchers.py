""" **Matchers**
#Kaze Matcher for binary classification - orb , kaze ,brief,fast
"""
import cv2
import numpy as np
from finalProject.utils.keyPoints.AlgoritamKeyPoints import SurfDetectKeyPoints
from finalProject.utils.keyPoints.AlgoritamKeyPoints import SiftDetectKeyPoints
from finalProject.utils.keyPoints.AlgoritamKeyPoints import ORBDetectKeyPoints
from finalProject.utils.keyPoints.AlgoritamKeyPoints import KazeDetectKeyPoints

def KazeMatcher(desc1, desc2):
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    nn_matches = matcher.knnMatch(desc1, desc2, 2)
    return nn_matches


"""# BF MATCHER"""


def findClosesHuman(human, myPeople, config: "config file"):
    keyTarget, descriptionTarget = SurfDetectKeyPoints(human["frame"])
    # TODO check with Liran if it's a different  DescriptionTarget here or not
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


def findSourceFeatures(human, myPeople, config: "config file"):
    keySourceSurf, descriptionSourceSurf = SurfDetectKeyPoints(human["frame"])
    keySourceSift, descriptionSourceSift = SiftDetectKeyPoints(human["frame"])
    keySourceOrb, descriptionSourceOrb = ORBDetectKeyPoints(human["frame"])
    keySourceKaze, descriptionSourceKaze = KazeDetectKeyPoints(human["frame"])

    if keySourceSurf is None or descriptionSourceSurf is None and keySourceSift is None or descriptionSourceSift is None:
        print("Both Surf and Sift has no key-points")
        return None  # dont have key points for this human
    maxMatch = []
    for p in myPeople:
        # remove trace frames
        if len(p.frames) > config["max_length_frames"]:
            p.history.extend(p.frames[0:len(p.frames) - config["max_length_frames"]])
            p.frames = p.frames[-config["max_length_frames"]:]

        MatchP = []
        for frame in p.frames:
            kpSurf, dpSurf = SurfDetectKeyPoints(frame)
            kpSift, dpSift = SiftDetectKeyPoints(frame)

            if kpSurf is None or dpSurf is None  and kpSift is None or dpSift is None:
                continue
            else:
                goodMatchSurf = FLANNMATCHER(descriptionSourceSurf, dpSurf, config["FlannMatcherThreshold"])
                goodMatchSift = FLANNMATCHER(descriptionSourceSift, dpSift, config["FlannMatcherThreshold"])
                maxMatches = max(len(goodMatchSurf), len(goodMatchSift))

                if  maxMatches == len(goodMatchSurf):
                    goodMatch = goodMatchSurf
                    maxKeySource = kpSurf
                    print("Surf has max match: ", len(goodMatch))
                    print("Sift has lower match: ", len(goodMatchSift))
                elif maxMatches == len(goodMatchSift):
                    goodMatch = goodMatchSift
                    maxKeySource = kpSift
                    print("Sift has max match: ", len(goodMatch))
                    print("Surf has lower match: ", len(goodMatchSurf))
            if len(goodMatch)== 0:
                acc = 0
            else:
                acc = len(goodMatch) / len(maxKeySource)
            MatchP.append(acc)
        if len(MatchP) > 0:
            MeanAcc = np.mean(MatchP)
        else:
            MeanAcc = 0
        maxMatch.append((p, MeanAcc))

    return maxMatch