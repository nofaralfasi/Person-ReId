import math


def SpeedEst(Loc1, Loc2):
    d_pixel = math.sqrt(
        math.pow(Loc2[0] - Loc1[0], 2) + math.pow(Loc2[1] - Loc1[1], 2))  # sqrt[(x2 - x1)^2 + (y2-y1)^2]
    ppm = 8.8
    d_meter = d_pixel / ppm
    fps = 18
    return d_meter * fps * 3.6


Loc1 = [100, 50, 50, 30]
Loc2 = [150, 50, 50, 30]

print(SpeedEst(Loc1, Loc2))
