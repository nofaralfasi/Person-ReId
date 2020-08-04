"""
pip install opencv-contrib-python==3.4.2.16
"""

import json
import pprint

import cv2
import matplotlib.pyplot as plt

from finalProject.classes.yolo import Yolo
from finalProject.utils.keyPoints.AlgoritamKeyPoints import sift_keypoints_detection
from finalProject.utils.matchers.Matchers import flann_matcher
from finalProject.utils.preprocessing.preprocess import read_frames_from_video, check_frames_exist
from finalProject.utils.tracking.TrackingByYolo import source_detection_by_yolo, tracking_by_yolo


def sobel(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    sobel_intensity = cv2.sqrt(cv2.addWeighted(cv2.pow(sobelx, 2.0),
                                               1.0, cv2.pow(sobely, 2.0), 1.0, 0.0))
    return sobel_intensity


def sobel_keypoints(image):
    sobelImage = sobel(image)
    # norm
    image8bit = cv2.normalize(sobelImage, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    k, d = sift_keypoints_detection(image8bit)
    return k, d, image8bit


def forward(frames):
    outputs = []
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = sobel(frame)
        key_des_image = sobel_keypoints(frame)
        outputs.append(key_des_image)

    return outputs


def cross_correction(key_des_image_source, key_des_image_target, threshold=0.5):
    (k1, d1, img1) = key_des_image_source
    (k2, d2, img2) = key_des_image_target
    match = flann_matcher(d1, d2, threshold=threshold)
    output = cv2.drawMatchesKnn(img1, k1, img2, k2, match, outImg=None)
    return output


def plotSquare(images, _titles, each_column=5):
    number_images = len(images)  # 10
    rows = number_images / each_column + 1
    index = 1

    fig = plt.figure(figsize=[9, 13])
    fig.tight_layout()

    for image, title in zip(images, _titles):
        plt.subplot(rows, each_column, index)
        plt.imshow(image)
        plt.title(title)
        index += 1

    plt.show()


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
        if not check_frames_exist(frameSource):
            print("problem with source video input")
            exit(0)

        mySource = source_detection_by_yolo(frameSource, yolo,
                                            is_video=config["source"]["isVideo"],
                                            config=config["source"])
        if mySource is None:
            print("fail to detect human on source video")
            exit(0)

        # target
        frameTarget = read_frames_from_video(config["target"])
        if not check_frames_exist(frameTarget):
            print("problem with target video input")
            exit(0)

        myTargets = tracking_by_yolo(frameTarget, yolo, is_video=config["target"]["isVideo"], config=config["target"])

        sources_output = forward(mySource.frames)  # array of outputs (key,des,image)
        targets_outputs = []  # array of targets , each target have array of frames
        for target in myTargets:
            target_output = forward(target.frames)  # array of frames
            targets_outputs.append(target_output)

        plots = []  # each frame

        for source_frame in sources_output:
            for target in targets_outputs:
                for target_frame in target:
                    plots.append(cross_correction(source_frame, target_frame))

        titles = ["test"] * 60
        plotSquare(plots, titles)
