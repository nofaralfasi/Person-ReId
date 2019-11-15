
"""# resize image"""
import cv2

def resizeImage(source,factor) : # size is tuple (w,h)
  return cv2.resize(source,None,fx=factor,fy=factor)


def Accuracy(kp,matches):
  return ( len(matches) / (len(kp)))

def ShowMatch(source,kp,target,kp2,matches):
  img3 = cv2.drawMatchesKnn(source,kp,target,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
  cv2.imshow("Match" ,img3)
  while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  cv2.destroyAllWindows()


"""# Quick started"""