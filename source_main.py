import cv2
import pprint
import os
from finalProject.classes.enumTypeKeyPoints import NamesAlgorithms
from finalProject.classes.yolo import Yolo
from finalProject.utils.drawing.draw import drawOnScatter
from finalProject.utils.keyPoints.AlgoritamKeyPoints import createDescriptorTarget
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

        # SourceDetectionByYolo returns a human
        mySource = SourceDetectionByYolo(frames, yolo, isVideo=config["isVideo"], config=config)

        if mySource is None:
            print("problem with source")
        else:
            descriptorSource = createDescriptorTarget([mySource])

            pp = pprint.PrettyPrinter(indent=4)

            pp.pprint(descriptorSource)

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

            for key in keys:
                if len(key[0]) > 0:
                    drawOnScatter(ax, key[0], key[1], label=key[2])

            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            ax.grid(True)
            plt.show()