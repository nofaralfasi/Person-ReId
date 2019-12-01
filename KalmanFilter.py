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
    def Estimate(self, coords):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coords[0])], [np.float32(coords[1])]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted


class Track:
    def __init__(self, index, coords):
        self.coords = coords
        self.index = index
        self.kf = KalmanFilter()
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
        vid = cv.VideoCapture('re-id/videos/storyLove.mp4')

        if vid.isOpened() == False:
            print('Cannot open input video')
            return

        width = int(vid.get(3))
        height = int(vid.get(4))

        myids = []
        while vid.isOpened():
            rc, frame = vid.read()

            if skipFrame < 1:
                rc, frame = vid.read()
                skipFrame += 1
                continue

            if rc:
                balls = yolo.forward(frame)

                if len(balls) > 0:
                    for b in balls:
                        cv.circle(frame, (int(b[0]), int(b[1])), 20, [255, 0, 0], 2, 8)

                myids = self.Update(balls, myids)

                print(len(myids))
                for myid in myids:
                    cv.circle(frame, (int(myid.coords[0]), int(myid.coords[1])), 20, [0, 0, 255], 2, 8)
                    cv.putText(frame, "Predicted id : " + str(myid.index)
                               , (int(myid.coords[0]), int(myid.coords[1] - 30)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])
                cv.imshow('Input', frame)

                if cv.waitKey(30) & 0xFF == ord('q'):
                    break

        vid.release()
        cv.destroyAllWindows()

    def Update(self, detections, myids):
        if len(detections) > 0:
            if len(myids) == 0:  # dont have yet ids but we detected
                # let assign all of them to myids
                for detect in detections:
                    track = Track(len(myids) + 1, detect)
                    myids.append(track)
            if len(myids) > 0:
                # so we have detections but we already have ids
                # lets detect them
                # create a cost function
                N = len(myids)
                M = len(detections)
                cost = np.zeros(shape=(N, M))  # Cost matrix
                for i in range(len(myids)):
                    for j in range(len(detections)):
                        try:
                            diff = myids[i].kf.Estimate(myids[i].coords) - detections[j]
                            distance = np.sqrt(diff[0][0] * diff[0][0] +
                                               diff[1][0] * diff[1][0])
                            cost[i][j] = distance
                        except:
                            pass

                # Let's average the squared ERROR
                cost = 0.5 * cost
                # Using Hungarian Algorithm assign the correct detected measurements
                # to predicted tracks
                row_ind, col_ind = linear_sum_assignment(cost)
                assignment = []
                for _ in range(N):
                    assignment.append(-1)

                for i in range(len(row_ind)):
                    assignment[row_ind[i]] = col_ind[i]

                # Identify tracks with no assignment, if any
                un_assigned_tracks = []
                for i in range(len(assignment)):
                    if assignment[i] != -1:
                        # check for cost distance threshold.
                        # If cost is very high then un_assign (delete) the track
                        print(cost[i][assignment[i]])
                        if cost[i][assignment[i]] > 1600:
                            assignment[i] = -1
                            un_assigned_tracks.append(i)
                        pass
                    else:
                        myids[i].skipped_frames += 1

                # If tracks are not detected for long time, remove them
                del_tracks = []
                for i in range(len(myids)):
                    if myids[i].skipped_frames > 10:
                        del_tracks.append(i)
                if len(del_tracks) > 0:  # only when skipped frame exceeds max
                    for id in del_tracks:
                        if id < len(myids):
                            del myids[id]
                            del assignment[id]
                        else:
                            print("ERROR: id is greater than length of tracks")
                # Now look for un_assigned detects
                un_assigned_detects = []
                for i in range(len(detections)):
                    if i not in assignment:
                        un_assigned_detects.append(i)

                if len(un_assigned_detects) != 0:
                    for i in range(len(un_assigned_detects)):
                        track = Track(len(myids) + 1, detections[un_assigned_detects[i]])
                        myids.append(track)

                # Update KalmanFilter state, lastResults and tracks trace
                for i in range(len(assignment)):
                    # self.tracks[i].KF.predict()

                    if assignment[i] != -1:
                        myids[i].skipped_frames = 0
                        myids[i].coords = myids[i].kf.Estimate(detections[assignment[i]])
                    else:
                        myids[i].coords = myids[i].kf.Estimate(np.array([[0], [0]]))

                return myids
        else:
            return []


# Main Function
def main():
    processImg = ProcessImage()
    processImg.DetectObjectLiran()


if __name__ == "__main__":
    main()

print('Program Completed!')
