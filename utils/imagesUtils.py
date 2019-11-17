"""# resize image"""
from builtins import dict

import cv2


def resizeImage(source, factor):  # size is tuple (w,h)
    return cv2.resize(source, None, fx=factor, fy=factor)


def Accuracy(kp, matches):
    return (len(matches) / (len(kp)))


def ShowMatch(source, kp, target, kp2, matches):

    draw_params = dict(
                       singlePointColor=None,
                       matchColor=(0, 255, 0),
                       flags=2)

    for i in kp:
      for j in kp2:
        img3 = cv2.drawMatchesKnn(source, i, target, j, matches[:10], None,**draw_params)
    cv2.imshow("Match", img3)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


"""# Quick started"""
