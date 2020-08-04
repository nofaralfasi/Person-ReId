"""
pip install opencv-contrib-python==3.4.2.16
"""

import json
import pprint

import cv2
import matplotlib.pyplot as plt

from finalProject.classes.yolo import Yolo
from finalProject.utils.keyPoints.AlgoritamKeyPoints import create_keypoints_descriptors
from finalProject.utils.matchers.Matchers import compute_accuracy_table
from finalProject.utils.preprocessing.preprocess import read_frames_from_video, is_frames_exists, reduce_noise, \
    removeRemovalColor
from finalProject.utils.tracking.TrackingByYolo import source_detection_by_yolo, tracking_by_yolo

if __name__ == "__main__":
    """# import images"""

    # init yolo
    yolo = Yolo()
    yolo.initYolo()
    pp = pprint.PrettyPrinter(indent=4)

    with open('./config.txt') as file_json:
        config = json.load(file_json)

        # source
        frameSource = read_frames_from_video(config["source"])
        if not is_frames_exists(frameSource):
            print("problem with source video input")
            exit(0)

        # pre processing reduce noise background
        if config["source"]["reduce_noise"]:
            frameSource = reduce_noise(frameSource)
        if not is_frames_exists(frameSource):
            print("problem with reduce noise source video input")
            exit(0)

        if config["source"]["removeRemovalColor"]:
            frameSource = removeRemovalColor(frameSource)

        # for frame in source_frames:
        #     cv2.imshow('removeRemovalColor frame', frame)
        #     keyboard = cv2.waitKey(30)
        #     if keyboard == 'q' or keyboard == 27:
        #         break

        mySource = source_detection_by_yolo(frameSource, yolo,
                                            is_video=config["source"]["isVideo"],
                                            config=config["source"])
        if mySource is None:
            print("fail to detect human on source video")
            exit(0)

        # source descriptor
        descriptorSource = create_keypoints_descriptors([mySource])

        # target
        frameTarget = read_frames_from_video(config["target"])
        if not is_frames_exists(frameTarget):
            print("problem with target video input")
            exit(0)

        if config["target"]["reduce_noise"]:
            frameTarget = reduce_noise(frameTarget)

        if not is_frames_exists(frameTarget):
            print("problem with target video input -reduce noise")
            exit(0)

        if config["target"]["removeRemovalColor"]:
            frameTarget = removeRemovalColor(frameTarget)

        myTargets = tracking_by_yolo(frameTarget, yolo, is_video=config["target"]["isVideo"], config=config["target"])

        if not is_frames_exists(myTargets):
            print("fail to detect humans on target video")
            exit(0)
        # target descriptor

        descriptorTarget = create_keypoints_descriptors(myTargets)

        # frameExampleTarget = target_descriptors[0][0]
        # frameExampleSource = source_descriptors[0][0]

        # drawFrameObject(frameExampleSource)
        # drawFrameObject(frameExampleTarget)

        acc_targets = compute_accuracy_table(descriptorSource, descriptorTarget)
        """
        acc_target look like :
         {
           id_0 : {
           maxAcc : double,
           # target : [arrayOfFrameObject]
           target_frames : FrameObject
           source_frames : FrameObject
           }
         }
        """
        target = "target", acc_targets[0]["target_frames"]["frame"]
        source = "source", acc_targets[0]["source_frames"]["frame"]

        target = target[1]
        source = source[1]

        cv2.destroyAllWindows()

        cv2.cvtColor(target, cv2.COLOR_RGB2GRAY, target)
        cv2.cvtColor(source, cv2.COLOR_RGB2GRAY, source)

        plt.subplot(121)
        plt.imshow(source)
        # plt.subplot(122)
        # plt.imshow(target)
        plt.show()

        # w, h, d = target.shape
        #
        #
        # # cv2.imshow("target with template match", target)
        # # cv2.imshow("source with template match", source)
        # # cv2.waitKey(0)
        # # Convert it to HSV
        # img1_hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
        # img2_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
        #
        # # Calculate the histogram and normalize it
        # hist_img1 = cv2.calcHist([img1_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        # cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        # hist_img2 = cv2.calcHist([img2_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        # cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        #
        # # find the metric value
        # metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
        # print(metric_val)

        # cv2.imshow("target with template match", target)
        # cv2.imshow("source with template match", source)
        # cv2.waitKey(0)
        #
        # # Apply template Matching
        # res = cv2.matchTemplate(source, target, cv2.TM_CCOEFF_NORMED)
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # top_left = max_loc

        # cv2.imshow(" source match", source)
        #
        # target = resizeImage(target, fy=2, fx=2)
        # cv2.imshow("target with template match", target)
        # cv2.waitKey(0)
