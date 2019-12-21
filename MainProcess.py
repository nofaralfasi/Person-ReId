import json
import pprint
from finalProject.classes.yolo import Yolo
from finalProject.utils.drawing.draw import drawFrameObject, drawTargetFinal
from finalProject.utils.keyPoints.AlgoritamKeyPoints import createDescriptorTarget
from finalProject.utils.matchers.Matchers import CompareBetweenTwoDescription
from finalProject.utils.preprocessing.preprocess import readFromInputVideoFrames, framesExists
from finalProject.utils.tracking.TrackingByYolo import SourceDetectionByYolo, TrackingByYolo

if __name__ == "__main__":
    """# import images"""

    # init yolo
    yolo = Yolo()
    yolo.initYolo()
    pp = pprint.PrettyPrinter(indent=4)

    with open('./config.txt') as file_json:
        config = json.load(file_json)

        # source
        frameSource = readFromInputVideoFrames(config["source"])
        if not framesExists(frameSource):
            print("problem with source video input")
            exit(0)
        mySource = SourceDetectionByYolo(frameSource, yolo,
                                         isVideo=config["source"]["isVideo"],
                                         config=config["source"])
        if mySource is None:
            print("fail to detect human on source video")
            exit(0)

        # source descriptor
        descriptorSource = createDescriptorTarget([mySource])

        # target
        frameTarget = readFromInputVideoFrames(config["target"])
        if not framesExists(frameTarget):
            print("problem with target video input")
            exit(0)

        myTargets = TrackingByYolo(frameTarget, yolo, isVideo=config["target"]["isVideo"], config=config["target"])

        if not framesExists(myTargets):
            print("fail to detect humans on target video")
            exit(0)
        # target descriptor
        descriptorTarget = createDescriptorTarget(myTargets)

        # frameExampleTarget = descriptorTarget[0][0]
        # frameExampleSource = descriptorSource[0][0]

        # drawFrameObject(frameExampleSource)
        # drawFrameObject(frameExampleTarget)

        acc_targets = CompareBetweenTwoDescription(descriptorSource, descriptorTarget)

        drawTargetFinal(acc_targets, descriptorSource)