import random
import itertools

import config
from utils import *
from parser import readFile

def calibrate_dist(packets, calibration, distance_resolution_m=0.002):
    nlasers = len(packets[0]["data"]["blocks"][0]["lasers"])

    vals = []
    for packet in packets:
        for b, block in enumerate(packet["data"]["blocks"]):
            for l, laser in enumerate(block["lasers"]):
                if laser["dist"] != 0:
                    vals.append(laser["dist"] * distance_resolution_m + calibration["lasers"][(b & 1) * nlasers + l]["dist_correction"])

    return vals


def getParams(data, calibration):
    vals = list(itertools.chain.from_iterable([calibrate_dist(d["packets"], calibration) for d in random.sample(data, 100)]))
    vals.sort()
    mu = sum(vals) / len(vals)

    head = 0
    tail = len(vals) - 1
    for _ in range(int(len(vals) / 20)):
        if mu - vals[head] >= vals[tail] - mu:
            head += 1
        else:
            tail -= 1

    theta = min(mu - vals[head], vals[tail] - mu)

    return mu, theta

if __name__ == '__main__':
    # data[number of sequences]["header"]["seq" / "stamp" / "frame_id"]
    # data[number of sequences]["packets"][number of packet data]["stamp" / "data"]
    # data[][][]["data"]["blocks"][number of blocks (=12)]["id" / "pos" / "lasers"]
    # data[][][][][][]["lasers"][number of lasers (=32)]["dist" / "intensity"]
    # data[][][]["data"]["stamp" / "type" / "value"]

    data = []
    for data_name in config.data_name:
        if file_exists(data_name):
            data += load_pickle(data_name)
        else:
            data += readFile(data_name, progress=True)
            save_pickle(data, data_name)

    print(len(data))

    # calibration["lasers"][number of lasers (=64)]["dist_correction" / "dist_correction_x" / "dist_correction_y"/
    #     "focal_distance" / "focal_slope" / "horiz_offset_correction" / "laser_id" / "min_intensity" /
    #     "rot_correction" / "vert_correction" / "vert_offset_correction"]
    # calibration["num_lasers"] = 64

    calibration = load_yaml(config.yaml_name)

    print(getParams(data, calibration))
