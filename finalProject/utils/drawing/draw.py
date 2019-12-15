import cv2
import matplotlib.pyplot as plt
import numpy as np
# from matplotlib import pyplot as plt
from finalProject.utils.drawing.common import draw_str

def DrawOnFrameMyIds(myids, frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 1
    # Line thickness of 2 px
    thicknessText = 1

    colorBlue = (255, 0, 0)
    colorRed = (100, 10, 200)

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
    img2 = frame
    thicknessRec = 2
    for index in affectedPeople:
        color = MyPeople[index].colorIndex
        # print(color)
        cv2.rectangle(frame, MyPeople[index].locations[-1][0], MyPeople[index].locations[-1][1],
                      color, thicknessRec)
        draw_str(frame, MyPeople[index].locations[-1][0], "id " + str(MyPeople[index].indexCount))

        img = cv2.imread('blank.jpg')
        img[MyPeople[index].locations[-1][0], MyPeople[index].locations[-1][1]]=[255,255,255]
        # cv2.imwrite('new.png',img)

        # Create a black image
        # img = np.zeros((512,512,3), np.uint8)
        # cv2.rectangle(img,(384,0),(510,128),(255,255,255),3)

        # rows,cols,channels = img2.shape
        # roi = img1[0:rows, 0:cols]

        # img = frame
        # frm = img[0:0, img.shape[0]:img.shape[1]]
        # img[0:0, img.shape[0]:img.shape[1]] =
        # mask = np.zeros(img.shape[:2],np.uint8)
        # bgdModel = np.zeros((1,65),np.float64)
        # fgdModel = np.zeros((1,65),np.float64)

        # rect = (50,50,450,290)
        # cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        # mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        # img = img*mask2[:,:,np.newaxis]
        # plt.imshow(img),plt.colorbar(),plt.show()

        # newmask is the mask image I manually labelled
        # newmask = cv2.imread('new.png',0)
        # wherever it is marked white (sure foreground), change mask=1
        # wherever it is marked black (sure background), change mask=0
        # mask[newmask == 0] = 0
        # mask[newmask == 255] = 1
        # mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
        # mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        # img = img*mask[:,:,np.newaxis]
        # plt.imshow(img),plt.colorbar(),plt.show()

def ShowPeopleTable(MyPeople, config: "configFile"):
    if len(MyPeople) == 0:
        print("no people found")
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