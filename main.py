"""
pip install opencv-contrib-python==3.4.2.16
"""

import json
import pprint

from finalProject.classes.yolo import Yolo
from finalProject.utils.drawing.draw import draw_final_results
from finalProject.utils.keyPoints.AlgoritamKeyPoints import create_keypoints_descriptors
from finalProject.utils.matchers.Matchers import compute_accuracy_table
from finalProject.utils.preprocessing.preprocess import read_frames_from_video, check_frames_exist, reduce_noise, removeRemovalColor
from finalProject.utils.tracking.TrackingByYolo import source_detection_by_yolo, tracking_by_yolo

if __name__ == "__main__":
    """# import images"""
    # init yolo
    yolo = Yolo()
    yolo.initYolo()
    pp = pprint.PrettyPrinter(indent=4)

    with open('./config.txt') as file_json:
        config = json.load(file_json)

        """ source video """
        source_frames = read_frames_from_video(config["source"])  # a list of all frames extracted from source video
        if not check_frames_exist(source_frames):  # if not len(source_frames) > 0
            print("problem with source video input")
            exit(0)

        # preprocessing reduce noise background
        if config["source"]["reduceNoise"]:
            source_frames = reduce_noise(source_frames)
        if not check_frames_exist(source_frames):
            print("problem with reduce noise source video input")
            exit(0)

        if config["source"]["removeRemovalColor"]:
            source_frames = removeRemovalColor(source_frames)

        # for frame in source_frames:
        #     cv2.imshow('removeRemovalColor frame', frame)
        #     keyboard = cv2.waitKey(30)
        #     if keyboard == 'q' or keyboard == 27:
        #         break

        source_person = source_detection_by_yolo(source_frames, yolo, is_video=config["source"]["isVideo"], config=config["source"])
        if source_person is None:
            print("fail to detect human on source video")
            exit(0)

        create_keypoints_descriptors([source_person])  # gets source descriptors to each frame

        """ target video """
        target_frames = read_frames_from_video(config["target"])
        if not check_frames_exist(target_frames):
            print("problem with target video input")
            exit(0)

        if config["target"]["reduceNoise"]:
            target_frames = reduce_noise(target_frames)
        if not check_frames_exist(target_frames):
            print("problem with target video input - in reduce noise")
            exit(0)

        if config["target"]["removeRemovalColor"]:
            target_frames = removeRemovalColor(target_frames)
        if not check_frames_exist(target_frames):
            print("problem with target video input - in remove color")
            exit(0)

        target_people = tracking_by_yolo(target_frames, yolo, is_video=config["target"]["isVideo"], config=config["target"])
        if not check_frames_exist(target_people):
            print("fail to detect humans on target video")
            exit(0)

        create_keypoints_descriptors(target_people)

        """
        acc_target looks like this:
         {
           id_0 : {
           maxAcc : double,
           # target : [arrayOfFrameObject]
           target_frames : FrameObject
           source_frames : FrameObject
           }
         }
        """
        acc_targets = compute_accuracy_table(source_person, target_people)

        draw_final_results(acc_targets, options=config["output"])
