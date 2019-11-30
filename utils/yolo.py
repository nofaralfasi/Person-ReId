import os
import numpy as np
import cv2
import math


def initYolo(weightPath: "yolov3.weights", configPath: "yolov3.cfg"):
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightPath)
    return net


def forward(net, image, labelPath: "coco.names"):
    LABELS = open(labelPath).read().strip().split("\n")
    ##initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")

    (H, W) = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

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
                croppingImages.append([[(x+w) // 2 , (y+h) // 2]])

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


def TrackingByYolo(sequences: [], net: "darkNet net", labelPath: "coco.names", isVideo: bool):
    myids = {}
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

            myTrackingObjectForward = forward(net, frame2, labelPath)

            # find ids from previous frame
            keysToRemove = []
            for _id in myids.values():
                # each crop find his id by center
                myNewTarget = findIdByCenter(myTrackingObjectForward, _id["centerHuman"])

                if myNewTarget is not None:
                    myNewTarget["centerHuman"] = getCenter(myNewTarget["box"])
                    myids[_id["_id"]]["centerHuman"] = myNewTarget["centerHuman"]
                    myids[_id["_id"]]["box"] = myNewTarget["box"]
                else:
                    # id was appear but now is missing
                    keysToRemove.append(_id["_id"])

            for keyRemove in keysToRemove:
                myids.pop(keyRemove)

            # print("missing ids ", len(myTrackingObjectForward))

            # create new ids
            for objectBox in myTrackingObjectForward:
                # each crop find his id by center
                newIdNumber = len(myids.keys()) + 1
                centerHuman = getCenter(objectBox["box"])

                myids[newIdNumber] = {"_id": newIdNumber, "centerHuman": centerHuman,
                                      "box": objectBox["box"]}

            frame2 = DrawOnFrameMyIds(myids, frame2)
            cv2.imshow('frame', frame2)
            k = cv2.waitKey(0) & 0xff
            if k == 27:
                break
