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


def findSourceFeatures(human, mySource, config: "config file"):
    keySourceSurf, descriptionSourceSurf = SurfDetectKeyPoints(human["frame"])
    keySourceSift, descriptionSourceSift = SiftDetectKeyPoints(human["frame"])
    keySourceOrb, descriptionSourceOrb = ORBDetectKeyPoints(human["frame"])
    keySourceKaze, descriptionSourceKaze = KazeDetectKeyPoints(human["frame"])

    maxKeyPoints = max(len(keySourceSurf), len(keySourceSift), len(keySourceOrb), len(keySourceKaze))
    maxDescriptions = max(len(descriptionSourceSurf), len(descriptionSourceSift), len(descriptionSourceOrb), len(descriptionSourceKaze))

    if maxKeyPoints == 0 or maxDescriptions == 0:
        print("Source has no key points or descriptions")
        return None  # dont have key points for this human

    maxMatch = []
    MatchP = []
    floatAcc = 0
    maxFloatKPSource = 0
    BinaryAcc = 0
    maxBinaryKPSource = 0

    for frame in mySource[0].frames:
        kpSurf, dpSurf = SurfDetectKeyPoints(frame)
        kpSift, dpSift = SiftDetectKeyPoints(frame)
        kpOrb, dpOrb = ORBDetectKeyPoints(frame)
        kpKaze, dpKaze = KazeDetectKeyPoints(frame)

        maxKP = max(len(kpSurf), len(kpSift), len(kpOrb), len(kpKaze))
        maxDes = max(len(dpSurf), len(dpSift), len(dpOrb), len(dpKaze))

        if maxKP == 0 or maxDes == 0:
            print("Not enough featurs or descriptions were found!")
            continue
        else:
            goodMatchSurf = FLANNMATCHER(descriptionSourceSurf, dpSurf, config["FlannMatcherThreshold"])
            goodMatchSift = FLANNMATCHER(descriptionSourceSift, dpSift, config["FlannMatcherThreshold"])
            goodMatchOrb = BFMatcher(descriptionSourceOrb, dpOrb, config["BFMatcherThreshold"])
            goodMatchKaze = KazeMatcher(descriptionSourceKaze, dpKaze)

            maxFloatMatches = max(len(goodMatchSurf), len(goodMatchSift))
            maxBinaryMatches = max(len(goodMatchOrb), len(goodMatchKaze))

            if maxFloatMatches == len(goodMatchSurf):
                # goodFloatMatch = goodMatchSurf
                # maxFloatKPSource = kpSurf
                print("Surf has max match: ", len(goodFloatMatch))
            elif maxFloatMatches == len(goodMatchSift):
                # goodFloatMatch = goodMatchSift
                # maxFloatKPSource = kpSift
                print("Sift has max match: ", len(goodFloatMatch))

            if maxBinaryMatches == len(goodMatchOrb):
                # goodBinaryMatch = goodMatchOrb
                # maxBinaryKPSource = kpOrb
                print("Orb has max match: ", len(goodBinaryMatch))
            elif maxBinaryMatches == len(goodMatchKaze):
                # goodBinaryMatch = goodMatchKaze
                # maxBinaryKPSource = kpKaze
                print("Kaze has max match: ", len(goodBinaryMatch))

        floatAcc = maxFloatMatches / len(maxFloatKPSource)
        BinaryAcc = maxBinaryMatches / len(maxBinaryKPSource)
        MatchP.append(floatAcc)
        MatchP.append(BinaryAcc)

    if len(MatchP) > 0:
        MeanAcc = np.mean(MatchP)
    else:
        MeanAcc = 0
    maxMatch.append((mySource[0], MeanAcc))

    return maxMatch
