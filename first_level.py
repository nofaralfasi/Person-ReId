import cv2
import pprint
import os
from finalProject.classes.enumTypeKeyPoints import NamesAlgorithms
from finalProject.classes.yolo import Yolo
from finalProject.utils.drawing.draw import drawOnScatter
from finalProject.utils.keyPoints.AlgoritamKeyPoints import createDescriptorSource
from finalProject.utils.tracking.TrackingByYolo import SourceDetectionByYolo
import matplotlib.pyplot as plt

import json

if __name__ == "__main__":
    """# import images"""

    with open('./config.txt') as file_json:

        config = json.load(file_json)

        frames = []

        if config["isVideo"]:
            cap = cv2.VideoCapture(config["inputVideo"])
            i = 0
            while i < config["skipRateFrameFromBeginning"]:
                ret, frame = cap.read()
                i += 1

            index = 0
            while ret:
                frames.append(frame)
                ret, frame = cap.read()
        else:
            path = config["inputVideo"]
            for (dirpath, dirnames, filenames) in os.walk(path):
                frames.extend(filenames)
                break

            frames.sort()
            frames = list(map(lambda file: path + "/" + file, frames))

        ## init yolo
        yolo = Yolo()
        yolo.initYolo()

        # cap = cv2.VideoCapture('vtest.avi')
        #
        # fgbg = cv2.createBackgroundSubtractorMOG()
        #
        # while(1):
        #     ret, frame = cap.read()
        #
        #     fgmask = fgbg.apply(frame)
        #
        #     cv2.imshow('frame',fgmask)
        #     k = cv2.waitKey(30) & 0xff
        #     if k == 27:
        #         break

        mySource = SourceDetectionByYolo(frames, yolo, isVideo=config["isVideo"], config=config)

        descriptorSource = createDescriptorSource(mySource)

        pp = pprint.PrettyPrinter(indent=4)

        # pp.pprint(descriptorSource)

        frameExmaple = descriptorSource[0][0]

        fig, ax = plt.subplots()

        frameExmaple["frame"] = cv2.cvtColor(frameExmaple["frame"], cv2.COLOR_BGR2RGB)

        ax.imshow(frameExmaple["frame"])

        keys = [
            (frameExmaple[NamesAlgorithms.KAZE.name]["keys"], 'tab:blue', NamesAlgorithms.KAZE.name),
            (frameExmaple[NamesAlgorithms.ORB.name]["keys"], 'tab:orange', NamesAlgorithms.ORB.name),
            (frameExmaple[NamesAlgorithms.SURF.name]["keys"], 'tab:green', NamesAlgorithms.SURF.name),
            (frameExmaple[NamesAlgorithms.SIFT.name]["keys"], 'tab:red', NamesAlgorithms.SIFT.name),
        ]

        # print("number of KAZE  key features: ",  len(keys[0]))
        # print("number of ORB  key features: ",  len(keys[1]))
        # print("number of SURF  key features: ",  len(keys[2]))
        # print("number of SIFT  key features: ",  len(keys[3]))

        for key in keys:
            if len(key[0]) > 0:
                drawOnScatter(ax, key[0], key[1], label=key[2])

        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.grid(True)
        plt.show()