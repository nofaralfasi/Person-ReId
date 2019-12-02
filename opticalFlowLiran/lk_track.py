#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import math
import video
from common import anorm2, draw_str
from time import clock

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)


class Track:
    def __init__(self, indexCount):
        self.indexCount = indexCount
        self.tr = []


class App:
    def __init__(self, video_src):
        self.track_len = 1000
        self.detect_interval = 1
        self.tracks = []
        self.cam = video.create_capture(video_src)
        self.frame_idx = 0
        self.indexCount = 0
        self.skipFrameOnStart = 15

    def runForwardVideo(self):
        while self.skipFrameOnStart > 0:
            self.cam.read()
            self.skipFrameOnStart -= 1

    def run(self):
        self.runForwardVideo()
        while True:
            _ret, frame = self.cam.read()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(self.tracks) > 0:
                # if we have track lets keep track those points
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr.tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    xt, yt = tr.tr[-1]
                    threshold = 0.9
                    if abs(x - xt) > threshold and abs(yt - y) > threshold:
                        tr.tr.append((x, y))
                        if len(tr.tr) > self.track_len:
                            del tr.tr[0]
                        new_tracks.append(tr)
                        cv.circle(vis, (x, y), 2, (0, 255, 0), -1)

                self.tracks = new_tracks

                cv.polylines(vis, [np.int32(tr.tr) for tr in self.tracks], False, (0, 255, 0))
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

                for tr in self.tracks:
                    draw_str(vis, (int(tr.tr[-1][0]), int(tr.tr[-1][1] - 30)), "id " + str(tr.indexCount))
                   # print(len(tr.tr))



            if self.frame_idx == 0:
                # first time
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                p = cv.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        trTemp = Track(self.indexCount)
                        self.indexCount += 1
                        trTemp.tr.append((x, y))
                        self.tracks.append(trTemp)
            else:
                # each 5 frame , looking for new object
                if self.frame_idx % self.detect_interval == 0:
                    mask = np.zeros_like(frame_gray)
                    mask[:] = 255
                    for x, y in [np.int32(tr.tr[-1]) for tr in self.tracks]:
                        cv.circle(mask, (x, y), 5, 0, -1)


                    p = cv.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            if not self.appendToCloseTr((x, y)):
                                trTemp = Track(self.indexCount)
                                self.indexCount += 1
                                trTemp.tr.append((x, y))
                                self.tracks.append(trTemp)

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv.imshow('lk_track', vis)

            ch = cv.waitKey(0)
            if ch == 27:
                break

    def appendToCloseTr(self, pt):
        if len(self.tracks) == 0:
            return False

        trackSelected = min(self.tracks, key=lambda item: self.distance(item.tr[-1], pt))

        d = (self.distance(trackSelected.tr[-1], pt))
        print(d)
        if d > 30:
            return False
        else:
            trackSelected.tr.append(pt)
            return True

    def distance(self, pt1, pt2):
        pn1 = np.array([pt1[0], pt1[1]])
        pn2 = np.array([pt2[0], pt2[1]])

        distance = math.sqrt((pn2[0] - pn1[0]) ** 2 + (pn2[1] - pn1[1]) ** 2)
        return distance


def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    App(video_src).run()
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
