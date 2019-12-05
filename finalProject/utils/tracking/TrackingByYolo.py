import cv2

from finalProject.classes.human import Human
from finalProject.utils.drawing.draw import DrawHumans, ShowPeopleTable
from finalProject.utils.matchers.Matchers import findClosesHuman
import copy


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

            drawFrame = copy.copy(frame2)
            if index == 0:
                # first time
                croppedImage = yolo.forward(frame2)
                for c in croppedImage:
                    # print(c)
                    if c["frame"].size:
                        human = Human(counterId)
                        counterId += 1
                        human.frames.append(c["frame"])
                        human.locations.append(c["location"])
                        myPeople.append(human)
            elif index > 0:
                croppedImage = yolo.forward(frame2)
                croppedImage = list(filter(lambda crop : crop["frame"].size,croppedImage))
                print("list of detection", len(croppedImage))
                for c in croppedImage:
                    if len(myPeople) > 0:
                        maxMatch = findClosesHuman(c, myPeople)
                        element = max(maxMatch, key=lambda item: item[1])
                        # cv2.imshow('target', c["frame"])
                        if element[1] > 0.2:  # score match
                            # cv2.imshow('HighScoreHuman', element[0].frames[-1])
                            # print('HighScoreHuman', element[1])
                            indexer = myPeople.index(element[0])
                            myPeople[indexer].frames.append(c["frame"])
                            myPeople[indexer].locations.append(c["location"])
                        # k = cv2.waitKey(10) & 0xff
                    else:
                        pass # TODO add him to myPeople

            DrawHumans(myPeople, drawFrame)
            # find ids from previous frame
            cv2.imshow('frame', drawFrame)
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

    for p in myPeople:
        print("number of frames in one person")
        print(len(p.frames))

    ShowPeopleTable(myPeople)
