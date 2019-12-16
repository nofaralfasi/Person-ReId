import cv2

from finalProject.classes.human import Human
from finalProject.utils.drawing.draw import DrawHumans, ShowPeopleTable, DrawSource
from finalProject.utils.matchers.Matchers import findClosesHuman
from finalProject.utils.matchers.Matchers import findSourceFeatures

import copy


def TrackingByYolo(sequences: [], yolo, isVideo: bool, config: "file"):
    myPeople = []
    counterId = 0
    frameRate = config["frameRate"]
    if config["videoFrameLength"] == -1:
        numOfFrames = len(sequences)
    else:
        numOfFrames = config["videoFrameLength"]

    if config["videoFrameLength"] > len(sequences):
        print("videoFrameLength larger then video")
        numOfFrames = len(sequences)

    if numOfFrames > 1:
        # start capture
        for index in range(0, numOfFrames, frameRate):
            affectedPeople = []

            print("frame {}".format(index))
            if isVideo:
                frame2 = sequences[index]
            else:
                frame2 = cv2.imread(sequences[index])

            drawFrame = copy.copy(frame2)
            if index == 0:
                # first time
                croppedImage = yolo.forward(frame2)
                croppedImage = list(filter(lambda crop: crop["frame"].size, croppedImage))
                for c in croppedImage:
                    # TODO check if they have any features
                    human = Human(counterId)
                    affectedPeople.append(counterId)
                    counterId += 1
                    human.frames.append(c["frame"])
                    human.locations.append(c["location"])
                    myPeople.append(human)
            elif index > 0:
                croppedImage = yolo.forward(frame2)
                croppedImage = list(filter(lambda crop: crop["frame"].size, croppedImage))
                # print("list of detection", len(croppedImage))
                for c in croppedImage:
                    if len(myPeople) > 0:
                        maxMatch = findClosesHuman(c, myPeople, config=config)
                        if maxMatch is None:
                            continue

                        element = max(maxMatch, key=lambda item: item[1])
                        # cv2.imshow('targetFromMovie', c["frame"])
                        print('scoreHumanFromMyPeople', element[1])
                        if element[1] > config["thresholdAppendToHuman"]:  # score match
                            # cv2.imshow('scoreHumanImageFromMyPeople', element[0].frames[-1])
                            indexer = myPeople.index(element[0])
                            affectedPeople.append(indexer)
                            myPeople[indexer].frames.append(c["frame"])
                            myPeople[indexer].locations.append(c["location"])
                        # k = cv2.waitKey(config["WaitKeySecond"]) & 0xff
                        # append new human in buckets
                        elif config["thresholdAppendNewHumanStart"] < element[1] < config["thresholdAppendNewHumanEnd"]:
                            human = Human(counterId)
                            affectedPeople.append(counterId)
                            counterId += 1
                            human.frames.append(c["frame"])
                            human.locations.append(c["location"])
                            myPeople.append(human)
                    else:
                        human = Human(counterId)
                        affectedPeople.append(counterId)
                        counterId += 1
                        human.frames.append(c["frame"])
                        human.locations.append(c["location"])
                        myPeople.append(human)

            DrawHumans(myPeople, drawFrame, affectedPeople)
            # find ids from previous frame
            cv2.imshow('frame', drawFrame)
            k = cv2.waitKey(config["WaitKeySecond"]) & 0xff
            if k == 27:
                break

    # print("number of people ", len(myPeople))
    # for index, p in enumerate(myPeople):
    #     print("number of frames in Person #", index)
    #     print(len(p.frames))
    #
    # ShowPeopleTable(myPeople, config=config)
    # print("done")

    return myPeople

# sequences is all frames related to the source
def SourceDetectionByYolo(sequences: [], yolo, isVideo: bool, config: "file"):
    frameCounter = 0
    frameRate = config["frameRate"]
    if config["videoFrameLength"] == -1:
        numOfFrames = len(sequences)
    else:
        numOfFrames = config["videoFrameLength"]

    if config["videoFrameLength"] > len(sequences):
        print("videoFrameLength larger then video")
        numOfFrames = len(sequences)

    if numOfFrames > 1:
        # start capture, looping on the frames, skipping the frameRate
        for index in range(0, numOfFrames, frameRate):
            print("frame {}".format(index))
            if isVideo:
                frame2 = sequences[index]
            else:
                frame2 = cv2.imread(sequences[index])

            drawFrame = copy.copy(frame2)
            croppedImage = yolo.forward(frame2)
            croppedImage = list(filter(lambda crop: crop["frame"].size, croppedImage))
            print("list of detection", len(croppedImage))

            # index is the frame number
            for c in croppedImage:
                if index == 0: # first frame
                    human = Human(frameCounter)
                    frameCounter += 1
                human.frames.append(c["frame"])
                human.locations.append(c["location"])
                if index > 0: # not the first frame
                    maxMatch = findSourceFeatures(c, human, config=config)
                    if maxMatch is None:
                        print("couldn't find features...")
                        continue
                    else:
                        element = max(maxMatch, key=lambda item: item[1])
                        print('My Source Score: ', element[1])
                        # TODO check if we want to save this frame?
                        # if element[1] > config["thresholdAppendToHuman"]:  # score match
                        #     human.frames.append(c["frame"])
                        #     human.locations.append(c["location"])
            DrawSource(human, drawFrame)
            # find ids from previous frame
            cv2.imshow('frame', drawFrame)
            k = cv2.waitKey(config["WaitKeySecond"]) & 0xff
            if k == 27:
                break
    else:
        print("The number of frames is less than one")

    # print("number of people ", len(myPeople))
    # for index, p in enumerate(myPeople):
    #     print("number of frames in Person #", index)
    #     print(len(p.frames))
    #
    # ShowPeopleTable(myPeople, config=config)
    # print("done")

    return human
