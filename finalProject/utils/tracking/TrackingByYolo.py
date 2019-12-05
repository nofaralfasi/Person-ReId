import cv2

from finalProject.classes.human import Human
from finalProject.utils.drawing.draw import DrawHumans
from finalProject.utils.matchers.Matchers import findClosesHuman


def TrackingByYolo(sequences: [], yolo, isVideo: bool):
    myPeople = []
    counterId = 0
    frameRate = 1
    numOfFrames = len(sequences)
    if numOfFrames > 1:
        # start capture
        for index in range(0, numOfFrames, frameRate):
            print("frame {}".format(index))
            if isVideo:
                frame2 = sequences[index]
            else:
                frame2 = cv2.imread(sequences[index])

            if index == 0:
                # first time
                croppedImage = yolo.forward(frame2)
                for c in croppedImage:
                    human = Human(counterId)
                    counterId += 1
                    human.frames.append(c["frame"])
                    human.locations.append(c["location"])
                    myPeople.append(human)
            elif index > 0:
                croppedImage = yolo.forward(frame2)
                for c in croppedImage:
                    maxMatch = findClosesHuman(c, myPeople)
                    print(maxMatch)
                    cv2.imshow('target', c["frame"])
                    k = cv2.waitKey(0) & 0xff

            # Todo add when index > 0 append to mypeople and compare keyPoints between two human
            DrawHumans(myPeople, frame2)
            # find ids from previous frame
            cv2.imshow('frame', frame2)
            k = cv2.waitKey(0) & 0xff
            if k == 27:
                break
