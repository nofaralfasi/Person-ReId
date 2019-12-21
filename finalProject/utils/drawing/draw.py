import cv2
import matplotlib.pyplot as plt
import numpy as np

from finalProject.classes.enumTypeKeyPoints import NamesAlgorithms
from finalProject.utils.drawing.common import draw_str


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


def DrawHumans(MyPeople, frame, affectedPeople):
    thicknessRec = 2
    for index in affectedPeople:
        color = MyPeople[index].colorIndex
        # print(color)
        cv2.rectangle(frame, MyPeople[index].locations[-1][0], MyPeople[index].locations[-1][1],
                      color, thicknessRec)
        draw_str(frame, MyPeople[index].locations[-1][0], "id " + str(MyPeople[index].indexCount))


def DrawSource(mySource, frame):
    thicknessRec = 2
    color = (255, 100, 150)
    cv2.rectangle(frame, mySource.locations[-1][0], mySource.locations[-1][1],
                  color, thicknessRec)
    draw_str(frame, mySource.locations[-1][0], "id " + str(mySource.indexCount))


def ShowPeopleTable(MyPeople, config: "configFile"):
    if len(MyPeople) == 0:
        print("no people were found!")
    else:
        if config["showHistory"]:
            photos = "history"
        else:
            photos = "frames"

        maxFramesHuman = max(MyPeople, key=lambda human: len(human.__getattribute__(photos)))

        rows = len(list(filter(lambda human: len(human.__getattribute__(photos)) > 0, MyPeople))) + 1

        cols = len(maxFramesHuman.__getattribute__(photos)) + 1

        print("rows ", rows)
        print("cols", cols)

        if rows > 0 and cols > 0:
            fig, ax = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True)
            for idx, human in enumerate(MyPeople):
                for jdx, frame in enumerate(human.__getattribute__(photos)):
                    print(idx, jdx)
                    ax[idx, jdx].imshow(frame)

            plt.show()


def show_images(images: list) -> None:
    n: int = len(images)
    f = plt.figure()
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(1, n, i + 1)
        # convert BGR to RGB
        # images[i] = cv2.cvtColor(images[i],images[i], cv2.COLOR_BGR2RGB)
        plt.imshow(images[i])
    plt.show(block=True)


def drawOnScatter(ax, keyPoints, color, label="none"):
    xyList = list(map(lambda keypoint: keypoint.pt, keyPoints))
    xl, yl = zip(*xyList)
    scale = 10
    ax.scatter(xl, yl, c=color, s=scale, label=label,
               alpha=0.8, edgecolors='none')


def drawFrameObject(frameObject, ax):
    frameObject["frame"] = cv2.cvtColor(frameObject["frame"], cv2.COLOR_BGR2RGB)

    ax.imshow(frameObject["frame"])

    keys = [
        (frameObject[NamesAlgorithms.KAZE.name]["keys"], 'tab:blue', NamesAlgorithms.KAZE.name),
        (frameObject[NamesAlgorithms.ORB.name]["keys"], 'tab:orange', NamesAlgorithms.ORB.name),
        (frameObject[NamesAlgorithms.SURF.name]["keys"], 'tab:green', NamesAlgorithms.SURF.name),
        (frameObject[NamesAlgorithms.SIFT.name]["keys"], 'tab:red', NamesAlgorithms.SIFT.name),
    ]

    for key in keys:
        if len(key[0]) > 0:
            drawOnScatter(ax, key[0], key[1], label=key[2])

    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.grid(True)

    #return ax
    # plt.show()


def drawTargetFinal(acc_targets):

    cols = len(acc_targets.keys())
    if len(acc_targets.keys()) == 1:
        cols += 1

    fig, axes = plt.subplots(nrows=2, ncols=cols, sharex=True, sharey=True)
    for key, target in acc_targets.items():
        # drawFrameObject(target["frameSource"], axes[0, key])
        # drawFrameObject(target["frameTarget"], axes[1, key])
        axes[0, key].imshow(target["frameSource"]["frame"])
        axes[1, key].imshow(target["frameTarget"]["frame"])
        axes[1, key].set_xlabel("Accuracy : " + str(target["maxAcc"]))

    plt.show()
