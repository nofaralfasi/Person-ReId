import cv2


class CroppedFrame:
    def __init__(self, frame_image):
        self.frame_image = frame_image
        self.gray_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
        self.frame_keypoints = {}
        self.frame_des = {}
        self.frame_parts = {'lowerbody': [],
                            'upperbody': [],
                            'face': []}
