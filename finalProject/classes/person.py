import numpy as np


class Person:
    def __init__(self, person_id):
        self.person_id = person_id
        self.frames = []
        self.missingFrames = 0
        self.locations = []
        self.history = []
        self.colorIndex = tuple(np.random.rand(3,)*255)
