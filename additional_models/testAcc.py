"""
pip install opencv-contrib-python==3.4.2.16
"""

import json
import pprint

from finalProject.classes.yolo import Yolo
from finalProject.utils.keyPoints.AlgoritamKeyPoints import surf_keypoints_detection, \
    kaze_keypoints_detection
from finalProject.utils.matchers.Matchers import kaze_matcher, flann_matcher
from finalProject.utils.preprocessing.preprocess import read_frames_from_video, is_frames_exists, reduce_noise
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
        frameSource = reduce_noise(frameSource)
        if not is_frames_exists(frameSource):
            print("problem with reduce noise source video input")
            exit(0)

        # for frame in source_frames:
        #     cv2.imshow('extracted frame', frame)
        #     keyboard = cv2.waitKey(30)
        #     if keyboard == 'q' or keyboard == 27:
        #         break

        mySource = source_detection_by_yolo(frameSource, yolo,
                                            is_video=config["source"]["isVideo"],
                                            config=config["source"])
        if mySource is None:
            print("fail to detect human on source video")
            exit(0)

        # target
        frameTarget = read_frames_from_video(config["target"])
        if not is_frames_exists(frameTarget):
            print("problem with target video input")
            exit(0)

        # pre processing reduce noise background
        frameTarget = reduce_noise(frameTarget)

        if not is_frames_exists(frameTarget):
            print("problem with target video input -reduce noise")
            exit(0)

        myTargets = tracking_by_yolo(frameTarget, yolo, is_video=config["target"]["isVideo"], config=config["target"])

        if not is_frames_exists(myTargets):
            print("fail to detect humans on target video")
            exit(0)
        # target descriptor

        ks, ds = kaze_keypoints_detection(mySource.frames[0])
        kt, dt = kaze_keypoints_detection(myTargets[0].frames[2])
        matches = kaze_matcher(ds, dt)
        print(len(ds))
        print(len(dt))
        print(len(ks))
        print(len(kt))
        print(len(matches))

        acc = len(matches) / min(len(ds), len(dt))
        print(acc)

        ks, ds = surf_keypoints_detection(mySource.frames[0])
        kt, dt = surf_keypoints_detection(myTargets[0].frames[2])
        matches = flann_matcher(ds, dt)
        acc = len(matches) / min(len(ds), len(dt))
        print(len(ds))
        print(len(dt))
        print(len(ks))
        print(len(kt))
        print(len(matches))
        print(acc)
