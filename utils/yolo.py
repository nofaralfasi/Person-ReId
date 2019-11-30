import os
import numpy as np
import cv2


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
                croppingImages.append({"box": [(x, y), (x + w, y + h)], "confidence": confidences[i]})

            # draw a bounding box rectangle and label on the image
            # color = [int(c) for c in COLORS[classIDs[i]]]
            # print(LABELS[classIDs[i]])
            # print(confidences[i])
    return croppingImages


def TrackingByYolo(sequences: [], net: "darkNet net", labelPath: "coco.names",isVideo: bool):
    colorBlue = (255, 0, 0)
    radius = 3
    thicknessCircle = -1
    thicknessRec = 2
    myids = []
    numOfFrames = len(sequences)
    if numOfFrames > 1:
        if isVideo:
            frame1 = sequences[0]
        else:
            frame1 = cv2.imread(sequences[0])

        myTrackingObject = forward(net, frame1, labelPath)

        for subImageDescribe in myTrackingObject:
            frame1 = cv2.rectangle(frame1, subImageDescribe["box"][0], subImageDescribe["box"][1], colorBlue, thicknessRec)
            centerHuman = ((subImageDescribe["box"][0][0] + subImageDescribe["box"][1][0]) // 2
                           , (subImageDescribe["box"][0][1] + subImageDescribe["box"][1][1]) // 2)

            frame1 = cv2.circle(frame1, centerHuman, radius, colorBlue, thicknessCircle)

        cv2.imshow('frame1', frame1)
        k = cv2.waitKey(0) & 0xff
