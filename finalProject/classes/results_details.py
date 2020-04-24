import cv2


class ResultsDetails:
    def __init__(self, config_data):
        self.input_name = config_data["inputVideo"]
        self.num_of_frames = 0
        self.reduce_noise = config_data["reduceNoise"]
        self.frame_rate = config_data["frameRate"]
        self.remove_color = config_data["removeRemovalColor"]
