import copy

import cv2

from finalProject.classes.cropped_frame import CroppedFrame
from finalProject.classes.person import Person
from finalProject.utils.drawing.draw import DrawHumans, DrawSource
from finalProject.utils.matchers.Matchers import find_closest_person


def tracking_by_yolo(frames: [], yolo, is_video: bool, config: "file"):
    """This function creates an array containing objects of all people that are in the target video,
    it assigns each person his frame-boxes from the video and other attributes,
    (this is regardless of the person in source video)."""

    people_list = []
    counter_id = 0
    frame_rate = config["frameRate"]

    if config["videoFrameLength"] == -1:
        num_of_frames = len(frames)
    else:
        num_of_frames = config["videoFrameLength"]

    if config["videoFrameLength"] > len(frames):
        print("videoFrameLength larger then video")
        num_of_frames = len(frames)

    if num_of_frames > 1:  # start capture
        for index in range(0, num_of_frames, frame_rate):
            affected_people = []  # a list of person_ids of all people that are in the current frame
            # print("frame {}".format(index))

            if is_video:
                frame2 = frames[index]
            else:
                frame2 = cv2.imread(frames[index])

            drawFrame = copy.copy(frame2)  # a copy list of the frame2 list
            if index == 0:  # first frame
                cropped_images = yolo.forward(frame2)  # returns all frame-boxes of people in the current frame
                cropped_images = list(filter(lambda crop: crop["frame"].size, cropped_images))  # filters frame-boxes
                for cropped_image in cropped_images:  # loop on frame-boxes
                    append_person_to_people(people_list, affected_people, counter_id, cropped_image)
                    counter_id += 1

            elif index > 0:
                cropped_images = yolo.forward(frame2)
                cropped_images = list(filter(lambda crop: crop["frame"].size, cropped_images))
                for cropped_image in cropped_images:  # determine which frame-box is related to which person
                    if len(people_list) > 0:
                        max_match = find_closest_person(cropped_image, people_list, config=config)  # returns a list of all people and their accuracy
                        if max_match is None:
                            continue  # skip iteration and continue on with the next iteration

                        max_maximum = max(max_match, key=lambda item: item[1])  # gets the person with max accuracy
                        if max_maximum[1] > config["thresholdAppendToHuman"]:  # accuracy is good enough to determine it's the same person
                            indexer = people_list.index(max_maximum[0])  # returns the index of this person with max accuracy in myPeople list
                            affected_people.append(indexer)  # adds this person to affected_people list
                            people_list[indexer].frames.append(CroppedFrame(cropped_image["frame"]))  # adds the box-frame to this person
                            people_list[indexer].locations.append(cropped_image["location"])  # adds his locations too

                        # TODO: consider calculating the thresholds according to the number of keypoints (of the person we're looking for)

                        elif config["thresholdAppendNewHumanStart"] < max_maximum[1] < config["thresholdAppendNewHumanEnd"]:  # if accuracy matches the values determine to create a new person
                            append_person_to_people(people_list, affected_people, counter_id, cropped_image)
                            counter_id += 1
                    else:  # creates a new person (this is the first person)
                        append_person_to_people(people_list, affected_people, counter_id, cropped_image)
                        counter_id += 1

            DrawHumans(people_list, drawFrame, affected_people)
            # find ids from previous frame
            if config["show"]:
                cv2.imshow('frame', drawFrame)
                k = cv2.waitKey(config["WaitKeySecond"]) & 0xff
                if k == 27:
                    break
    return people_list


def append_person_to_people(people_list, affected_people_ids, person_id, cropped_image):
    person = Person(person_id)
    affected_people_ids.append(person_id)
    person.frames.append(CroppedFrame(cropped_image["frame"]))
    person.locations.append(cropped_image["location"])
    people_list.append(person)


def source_detection_by_yolo(source_frames: [], yolo, is_video: bool, config: "file"):
    person = Person(0)
    frame_rate = config["frameRate"]
    if config["videoFrameLength"] == -1:
        num_of_frames = len(source_frames)
    else:
        num_of_frames = config["videoFrameLength"]

    if config["videoFrameLength"] > len(source_frames):
        print("videoFrameLength larger then video")
        num_of_frames = len(source_frames)

    if num_of_frames > 1:
        for index in range(0, num_of_frames, frame_rate):  # start capture, looping on the frames, skipping the frameRate
            # print("frame {}".format(index))
            if is_video:
                current_frame = source_frames[index]  # index is the frame number
            else:
                current_frame = cv2.imread(source_frames[index])

            drawFrame = copy.copy(current_frame)
            cropped_frames = yolo.forward(current_frame)
            cropped_frames = list(filter(lambda crop: crop["frame"].size, cropped_frames))

            if len(cropped_frames) > 1:
                print("On source found two people, must be one person")
                return None

            for cropped_frame in cropped_frames:
                person.frames.append(CroppedFrame(cropped_frame["frame"]))
                person.locations.append(cropped_frame["location"])

            if len(person.frames) > 0:
                DrawSource(person, drawFrame)

            if config["show"]:
                cv2.imshow('frame', drawFrame)
                k = cv2.waitKey(config["WaitKeySecond"]) & 0xff
                if k == 27:
                    break
    else:
        print("The number of frames is less than one")

    return person
