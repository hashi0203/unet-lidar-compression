import random
import itertools
import numpy as np
import datetime

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

import matplotlib.pyplot as plt

def getParams(data, calibration):
    vals = list(itertools.chain.from_iterable([calibrate_dist(d["packets"], calibration) for d in random.sample(data, 100)]))
    mu = np.mean(vals)
    theta = np.std(vals)

    t = datetime.datetime.now().strftime('%m%d-%H%M')
    if not os.path.isdir(config.GRAPH_PATH):
        os.mkdir(config.GRAPH_PATH)
    PARAM_FILE = os.path.join(config.GRAPH_PATH, 'param-%s.png' % t)
    plt.figure()
    plt.hist(vals, bins=50, range=(0,50))
    plt.savefig(PARAM_FILE)

    return mu, theta

if __name__ == '__main__':
    # raw_data[number of sequences]["header"]["seq" / "stamp" / "frame_id"]
    # raw_data[number of sequences]["packets"][number of packet data]["stamp" / "data"]
    # raw_data[][][]["data"]["blocks"][number of blocks (=12)]["id" / "pos" / "lasers"]
    # raw_data[][][][][][]["lasers"][number of lasers (=32)]["dist" / "intensity"]
    # raw_data[][][]["data"]["stamp" / "type" / "value"]

    data = []
    for data_name in config.data_name:
        if file_exists(data_name, 'raw'):
            data += load_pickle(data_name, 'raw')
        else:
            raw_data = readFile(data_name, progress=True)
            save_pickle(raw_data, data_name, 'raw')
            data += raw_data

    print(len(data))

    # calibration["lasers"][number of lasers (=64)]["dist_correction" / "dist_correction_x" / "dist_correction_y"/
    #     "focal_distance" / "focal_slope" / "horiz_offset_correction" / "laser_id" / "min_intensity" /
    #     "rot_correction" / "vert_correction" / "vert_offset_correction"]
    # calibration["num_lasers"] = 64

    calibration = load_yaml(config.yaml_name)

    print(getParams(data, calibration))
