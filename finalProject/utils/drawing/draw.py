import cv2
import matplotlib.pyplot as plt
import numpy as np


def DrawOnFrameMyIds(myids, frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 1
    # Line thickness of 2 px
    thicknessText = 1

    colorBlue = (255, 0, 0)
    colorRed = (0, 0, 255)

    radius = 3
    thicknessCircle = -1
    thicknessRec = 2

    for _id in myids.values():
        frame = cv2.rectangle(frame, myids[_id["_id"]]["box"][0], myids[_id["_id"]]["box"][1],
                              colorBlue, thicknessRec)
        frame = cv2.circle(frame, myids[_id["_id"]]["centerHuman"], radius, colorRed, thicknessCircle)
        frame = cv2.putText(frame, 'ID:' + str(_id["_id"]), (myids[_id["_id"]]["centerHuman"][0]
                                                             , myids[_id["_id"]]["centerHuman"][1] - 50), font,
                            fontScale, (255, 0, 0), thicknessText, cv2.LINE_AA)
    return frame


def DrawHumans(MyPeople, frame):
    colorBlue = (255, 0, 0)
    thicknessRec = 2
    for human in MyPeople:
        cv2.rectangle(frame, human.locations[-1][0], human.locations[-1][1],
                      colorBlue, thicknessRec)


def ShowPeopleTable(MyPeople):
    maxFramesHuman= max(MyPeople, key=lambda human: len(human.history))
    fig, ax = plt.subplots(nrows=len(MyPeople)+1, ncols=len(maxFramesHuman.history)+1, sharex=True, sharey=True, )
    for idx,human in enumerate(MyPeople):
        for jdx,frame in enumerate(human.history):
            print(idx,jdx)
            ax[idx,jdx].imshow(frame)

    plt.show()


def show_images(images: list) -> None:
    n: int = len(images)
    f = plt.figure()
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(1, n, i + 1)
        plt.imshow(images[i])
    plt.show(block=True)
