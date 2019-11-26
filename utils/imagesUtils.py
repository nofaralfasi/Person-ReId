"""# resize image"""
from builtins import dict
import random as rnd
import cv2


def resizeImage(source, fx, fy):  # size is tuple (w,h)
    return cv2.resize(source, None, fx=fx, fy=fy)


def Accuracy(kp, matches):
    return (len(matches) / (len(kp)))


def ShowMatch(source, kp, target, kp2, matches):
    draw_params = dict(
        singlePointColor=None,
        matchColor=(255, 0, 0),
        flags=2)

    print("matches length {}".format(len(matches)))
    img3 = cv2.drawMatchesKnn(source, kp, target, kp2, matches[:80], None, **draw_params)
    cv2.imshow("Match", img3)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


"""# Quick started"""
