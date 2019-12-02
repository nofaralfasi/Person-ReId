import os
import numpy as np
import cv2
import math


class Yolo:
    def __init__(self, weightPath="yolo-object-detection/yolo-coco/yolov3.weights"
                 , configPath="yolo-object-detection/yolo-coco/yolov3.cfg",
                 labelPath='yolo-object-detection/yolo-coco/coco.names'):
        self.configPath = configPath
        self.weightPath = weightPath
        self.labelPath = labelPath
        self.net = None

    def initYolo(self):
        print("[INFO] loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightPath)

    def forward(self, image):
        LABELS = open(self.labelPath).read().strip().split("\n")

        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                                   dtype="uint8")

        (H, W) = image.shape[:2]
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(ln)

        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > 0.5:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        # ensure at least one detection exists

        croppingImages = []
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                if LABELS[classIDs[i]] == 'person':
                    # croppingImages.append({"box": [(x, y), (x + w, y + h)], "confidence": confidences[i]})
                    croppingImages.append([(x, y), (x+w, y+h)])

                # draw a bounding box rectangle and label on the image
                # color = [int(c) for c in COLORS[classIDs[i]]]
                # print(LABELS[classIDs[i]])
                # print(confidences[i])
        return croppingImages


def distance(pt1, pt2):
    d = math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
    return d


def getCenter(box):
    return (box[1][0] + box[0][0]) // 2, (box[1][1] + box[0][1]) // 2


def findIdByCenter(objectsTargets, OldPositionCenter):
    if len(objectsTargets) > 0:
        d = min(objectsTargets, key=lambda dis: distance(getCenter(dis["box"]), OldPositionCenter))
        objectsTargets.remove(d)
        return d
    else:
        return None  # id is missing


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


class Human:
    def __init__(self, indexCount):
        self.indexCount = indexCount
        self.frames = []
        self.missingFrames = 0


def DrawHumans(humans, frame):
    colorBlue = (255, 0, 0)
    thicknessRec = 2
    for h in humans:
        cv2.rectangle(frame, h.frames[-1][0], h.frames[-1][1],
                      colorBlue, thicknessRec)


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
                    human.frames.append(c)
                    myPeople.append(human)
            # Todo add when index > 0 append to mypeople and compare keyPoints between two human
            DrawHumans(myPeople, frame2)
            # find ids from previous frame
            cv2.imshow('frame', frame2)
            k = cv2.waitKey(0) & 0xff
            if k == 27:
                break
