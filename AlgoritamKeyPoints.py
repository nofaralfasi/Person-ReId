"""# Sift Algoritam"""
import cv2


def KeyPointsBinary(img, Threshold):
    kpOrb, desOrb = ORBDetectKeyPoints(img, Threshold)
    kpKaze, desKaze = KazeDetectKeyPoints(img)
    return [(kpOrb,desOrb), (kpKaze,desKaze)]


def KeyPointsFloat(img, Threshold):
    kpSurf, desSurf = SuftDetectKeyPoints(img, Threshold)
    kpSift, desSift = SiftDetectKeyPoints(img, Threshold)

    return [(kpSurf,desSurf), (kpSift,desSift)]


def SiftDetectKeyPoints(img, Threshold):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return (kp, des)


"""# Surf Algoritam"""


def SuftDetectKeyPoints(img, Threshold):
    suft = cv2.xfeatures2d.SURF_create()
    kp, des = suft.detectAndCompute(img, None)
    return (kp, des)


"""#orb algoritam"""


def ORBDetectKeyPoints(img, Threshold):
    # Initiate STAR detector
    orb = cv2.ORB_create(nfeatures=Threshold)  # find the keypoints with ORB
    kp = orb.detect(img, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    return kp, des


"""# Kaze algoritam"""


def KazeDetectKeyPoints(img):
    akaze = cv2.AKAZE_create()
    kp, des = akaze.detectAndCompute(img, None)
    return kp, des
