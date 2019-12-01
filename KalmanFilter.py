# --------------------------------------------------------------------
# Implements Ball motion prediction using Kalman Filter
#
# Author: Sriram Emarose [sriram.emarose@gmail.com]
#
#
#
# --------------------------------------------------------------------

import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.yolo import Yolo


# Instantiate OCV kalman filter
class KalmanFilter:
    kf = cv.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    lastResult = np.array([[0], [255]])

    #
    def Estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted

    def predict(self):
        predicted = self.kf.predict()
        return predicted


class Track:
    def __init__(self, index, coords):
        self.coords = coords
        self.index = index
        self.kf = KalmanFilter()
        self.predictions = self.kf.Estimate(coords[0], coords[1])
        self.skipped_frames = 0


# Performs required image processing to get ball coordinated in the video
class ProcessImage:

    def DetectObject(self):
        yolo = Yolo()
        yolo.initYolo()  # upload weights to ram
        track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                        (0, 255, 255), (255, 0, 255), (255, 127, 255),
                        (127, 0, 255), (127, 0, 127)]

        kfObject = KalmanFilter()

        skipFrame = 0
        vid = cv.VideoCapture('re-id/videos/How People Walk.mp4')

        if vid.isOpened() == False:
            print('Cannot open input video')
            return

        width = int(vid.get(3))
        height = int(vid.get(4))

        # Create Kalman Filter Object
        # kfObj = KalmanFilter()
        predictedCoords = np.zeros((2, 1), np.float32)

        while vid.isOpened():
            rc, frame = vid.read()

            if skipFrame < 1300:
                rc, frame = vid.read()
                skipFrame += 1
                continue

            if rc:
                balls = yolo.forward(frame)

                if len(balls) > 0:
                    for ball in balls:
                        # create track
                        predictedCoords = kfObject.Estimate(ball[0], ball[1])

                        ballX, ballY = ball

                        print(ballX, ballY)
                        print(predictedCoords)

                        # Draw Actual coords from segmentation
                        cv.circle(frame, (int(ballX), int(ballY)), 20, [0, 0, 255], 2, 8)
                        cv.line(frame, (int(ballX), int(ballY + 20)), (int(ballX + 50), int(ballY + 20)),
                                [100, 100, 255],
                                2, 8)
                        cv.putText(frame, "Actual", (int(ballX + 50), int(ballY + 20)), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                                   [50, 200, 250])

                        # Draw Kalman Filter Predicted output
                        cv.circle(frame, (predictedCoords[0], predictedCoords[1]), 20, [0, 255, 255], 2, 8)
                        cv.line(frame, (predictedCoords[0] + 16, predictedCoords[1] - 15),
                                (predictedCoords[0] + 50, predictedCoords[1] - 30), [100, 10, 255], 2, 8)
                        cv.putText(frame, "Predicted", (int(predictedCoords[0] + 50), int(predictedCoords[1] - 30)),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])

                cv.imshow('Input', frame)

                if cv.waitKey(30) & 0xFF == ord('q'):
                    break

        vid.release()
        cv.destroyAllWindows()

    def DetectObjectLiran(self):
        yolo = Yolo()
        yolo.initYolo()  # upload weights to ram
        track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                        (0, 255, 255), (255, 0, 255), (255, 127, 255),
                        (127, 0, 255), (127, 0, 127)]

        # kfObject = KalmanFilter()

        skipFrame = 0
        vid = cv.VideoCapture('re-id/videos/How People Walk.mp4')

        if vid.isOpened() == False:
            print('Cannot open input video')
            return

        width = int(vid.get(3))
        height = int(vid.get(4))

        # Create Kalman Filter Object
        # kfObj = KalmanFilter()
        # predictedCoords = np.zeros((2, 1), np.float32)

        trackingList = []
        while vid.isOpened():
            rc, frame = vid.read()

            if skipFrame < 1300:
                rc, frame = vid.read()
                skipFrame += 1
                continue

            if rc:
                balls = yolo.forward(frame)

                if len(balls) > 0:
                    if len(trackingList) == 0:
                        # we have balls and we dont tracking yet ,lets assign them
                        for b in balls:
                            indexCounter = len(trackingList) + 1
                            track = Track(indexCounter, b)
                            trackingList.append(track)

                # for t in trackingList:
                #     frame = cv.circle(frame, (int(t.coords[0]), int(t.coords[1])), 20, [0, 0, 255], 2, 8)

                cv.imshow('Input', frame)

                if cv.waitKey(30) & 0xFF == ord('q'):
                    break

        vid.release()
        cv.destroyAllWindows()

    def Assigns(self, balls, trackingList):

        N = len(trackingList)
        M = len(balls)
        cost = np.zeros(shape=(N, M))  # Cost matrix
        for i in range(len(trackingList)):
            for j in range(len(balls)):
                try:
                    diff = trackingList[i].predictions - balls[j]
                    distance = np.sqrt(diff[0][0] * diff[0][0] +
                                       diff[1][0] * diff[1][0])
                    cost[i][j] = distance
                except:
                    pass

        # Let's average the squared ERROR
        cost = (0.5) * cost
        # Using Hungarian Algorithm assign the correct detected measurements
        # to predicted tracks
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)

        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if assignment[i] != -1:
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                # print(cost[i][assignment[i]])
                if cost[i][assignment[i]] > 160:
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                pass
            else:
                trackingList[i].skipped_frames += 1

        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(trackingList)):
            if trackingList[i].skipped_frames > 100:
                del_tracks.append(i)
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in del_tracks:
                if id < len(trackingList):
                    del trackingList[id]
                    del assignment[id]
                else:
                    print("ERROR: id is greater than length of tracks")

        un_assigned_detects = []
        for i in range(len(balls)):
            if i not in assignment:
                un_assigned_detects.append(i)

        # Start new tracks
        if len(un_assigned_detects) != 0:
            for i in range(len(un_assigned_detects)):
                track = Track(len(trackingList) + 1, balls[un_assigned_detects[i]])
                trackingList.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            trackingList[i].kf.predict()

            if assignment[i] != -1:
                trackingList[i].skipped_frames = 0
                trackingList[i].predictions = trackingList[i].kf.Estimate(balls[assignment[i]][0],
                                                                          balls[assignment[i]][1])
            else:
                trackingList[i].predictions = trackingList[i].kf.Estimate(0, 0)

            # if len(trackingList[i].trace) > self.max_trace_length:
            #     for j in range(len(trackingList[i].trace) -
            #                    self.max_trace_length):
            #         del trackingList[i].trace[j]

            # trackingList[i].trace.append(trackingList[i].prediction)
            trackingList[i].kf.lastResult = trackingList[i].predictions


# Main Function
def main():
    processImg = ProcessImage()
    processImg.DetectObjectLiran()


if __name__ == "__main__":
    main()

print('Program Completed!')
