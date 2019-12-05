import os
import numpy as np
import cv2
import math


class Human:
    def __init__(self, indexCount):
        self.indexCount = indexCount
        self.frames = []
        self.missingFrames = 0
