""" **Matchers**

#Kaze Matcher for binary classification - orb , kaze ,brief,fast
"""
import cv2

from finalProject.utils.keyPoints.AlgoritamKeyPoints import SuftDetectKeyPoints


def KazeMatcher(desc1, desc2):
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    nn_matches = matcher.knnMatch(desc1, desc2, 2)
    return nn_matches


"""# BF MATCHER"""


def findClosesHuman(human, myPeople, max_length_frames=3):
    keyTarget, DescriptionTarget = SuftDetectKeyPoints(human["frame"])
    maxMatch = []

    for p in myPeople:
        if len(p.frames) > max_length_frames:
            p.history.extend(p.frames[0:len(p.frames)-max_length_frames])
            p.frames = p.frames[-max_length_frames:]

        MatchP = []
        for frame in p.frames:
            kp, dp = SuftDetectKeyPoints(frame)
            if len(dp) > 0 and len(DescriptionTarget) > 0:
                goodMatch = FLANNMATCHER(DescriptionTarget, dp, 0.9)
            if len(keyTarget) == 0:
                acc = 0
            else:
                acc = len(goodMatch) / len(keyTarget)
            MatchP.append(acc)
        MaxAcc = max(MatchP)
        maxMatch.append((p, MaxAcc))

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
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    good = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < threshold * n.distance:
            good.append([m])
    return good
