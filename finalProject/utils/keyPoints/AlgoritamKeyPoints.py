"""# Sift Algoritam"""
import cv2

from finalProject.classes.enumTypeKeyPoints import NamesAlgorithms


def KeyPointsBinary(img, Threshold):
    kpOrb, desOrb = ORBDetectKeyPoints(img, Threshold)
    kpKaze, desKaze = KazeDetectKeyPoints(img)
    return [(kpOrb, desOrb), (kpKaze, desKaze)]


def KeyPointsFloat(img, Threshold):
    kpSurf, desSurf = SurfDetectKeyPoints(img, Threshold)
    kpSift, desSift = SiftDetectKeyPoints(img, Threshold)

    return [(kpSurf, desSurf), (kpSift, desSift)]


def SiftDetectKeyPoints(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


"""# Surf Algoritam"""


def SurfDetectKeyPoints(img, Threshold=0.5):
    suft = cv2.xfeatures2d.SURF_create()
    kp, des = suft.detectAndCompute(img, None)
    return kp, des


"""#orb algoritam"""


def ORBDetectKeyPoints(img, n_features=400):
    # Initiate STAR detector
    orb = cv2.ORB_create(nfeatures=n_features)  # find the keypoints with ORB
    kp = orb.detect(img, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    return kp, des


"""# Kaze algoritam"""


def KazeDetectKeyPoints(img):
    akaze = cv2.AKAZE_create()
    kp, des = akaze.detectAndCompute(img, None)
    return kp, des


def CalculationKeyPoint(image, keyPointFunction):
    return keyPointFunction(image)


def appendToFrameObject(keys, descriptions, label, frameObject):
    if keys is None or descriptions is None or len(keys) == 0 or len(descriptions) == 0:
        frameObject[label] = {"keys": [], "des": []}
    else:
        frameObject[label] = {"keys": keys, "des": descriptions}


def createDescriptorTarget(myTarget):
    descriptor = {}

    for target in myTarget:
        descriptor[target.indexCount] = []

        for frame in target.frames:
            kOrb, desOrb = CalculationKeyPoint(frame, ORBDetectKeyPoints)
            kKaze, desKaze = CalculationKeyPoint(frame, KazeDetectKeyPoints)
            kSift, desSift = CalculationKeyPoint(frame, SiftDetectKeyPoints)
            kSurf, desSurf = CalculationKeyPoint(frame, SurfDetectKeyPoints)

            frameObject = {
                "frame": frame,
            }
            appendToFrameObject(kOrb, desOrb, NamesAlgorithms.ORB.name, frameObject)
            appendToFrameObject(kKaze, desKaze, NamesAlgorithms.KAZE.name, frameObject)
            appendToFrameObject(kSift, desSift, NamesAlgorithms.SIFT.name, frameObject)
            appendToFrameObject(kSurf, desSurf, NamesAlgorithms.SURF.name, frameObject)

            descriptor[target.indexCount].append(frameObject)

    return descriptor
