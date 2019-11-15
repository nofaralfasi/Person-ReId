
"""# Sift Algoritam"""
import cv2

def SiftDetectKeyPoints(img,Threshold):
  sift = cv2.xfeatures2d.SIFT_create()
  kp, des = sift.detectAndCompute(img,None)
  return (kp,des)

"""# Surf Algoritam"""

def SuftDetectKeyPoints(img,Threshold):
  suft = cv2.xfeatures2d.SURF_create()
  kp, des = suft.detectAndCompute(img,None)
  return (kp,des)

"""#orb algoritam"""

def ORBDetectKeyPoints(img,Threshold):
  # Initiate STAR detector
  orb = cv2.ORB_create(nfeatures=Threshold)  # find the keypoints with ORB
  kp = orb.detect(img,None)
  # compute the descriptors with ORB
  kp, des = orb.compute(img, kp)
  return kp,des

"""# Kaze algoritam"""

def KazeDetectKeyPoints(img):
  akaze = cv2.AKAZE_create()
  kp, des = akaze.detectAndCompute(img, None)
  return kp,des
