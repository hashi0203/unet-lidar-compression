from tqdm import tqdm

import config
from utils import *
from parser import readFile

def preprocess(packets, calibration, distance_resolution_m=0.002):
    npackets = len(packets)
    nblocks = len(packets[0]["data"]["blocks"])
    nlasers = len(packets[0]["data"]["blocks"][0]["lasers"])

    mu = config.mu # 100 フレームの平均
    theta = config.theta # 100 フレームの標準偏差の半分

    img = [[packets[p]["data"]["blocks"][(b << 1) + (l // nlasers)]["lasers"][l % nlasers]["dist"] for b in range((nblocks >> 1)) for p in range(npackets)] for l in range(nlasers << 1)]
    for i in range(nlasers << 1):
        for j in range(npackets * (nblocks >> 1)):
            if img[i][j] == 0:
                if j > 1:
                    img[i][j] = img[i][j-1]
                elif i > 1:
                    img[i][j] = img[i-1][j]
                else:
                    img[i][j] = (img[i][j] * distance_resolution_m + calibration["lasers"][i]["dist_correction"] - mu) / theta
            else:
                img[i][j] = (img[i][j] * distance_resolution_m + calibration["lasers"][i]["dist_correction"] - mu) / theta

    return img

def raw2img(data, calibration, progress=False):
    if progress:
        data = tqdm(data)
        data.set_description("Converting data into 2D image")
    return [preprocess(d["packets"], calibration) for d in data]

def calibrate(packets, calibration, distance_resolution_m=0.002):
    npackets = len(packets)
    nblocks = len(packets[0]["data"]["blocks"])
    nlasers = len(packets[0]["data"]["blocks"][0]["lasers"])

    mu = config.mu # 100 フレームの平均
    theta = config.theta # 100 フレームの標準偏差の半分

    calibrated = [[(packets[p]["data"]["blocks"][(b << 1) + (l // nlasers)]["lasers"][l % nlasers]["dist"] * distance_resolution_m + calibration["lasers"][l]["dist_correction"] - mu) / theta
                    for b in range((nblocks >> 1)) for p in range(npackets)] for l in range(nlasers << 1)]

    return calibrated

def raw2calibrated(data, calibration, progress=False):
    if progress:
        data = tqdm(data)
        data.set_description("Calibrating data")
    return [calibrate(d["packets"], calibration) for d in data]


if __name__ == '__main__':
    # raw_data[number of sequences]["header"]["seq" / "stamp" / "frame_id"]
    # raw_data[number of sequences]["packets"][number of packet data]["stamp" / "data"]
    # raw_data[][][]["data"]["blocks"][number of blocks (=12)]["id" / "pos" / "lasers"]
    # raw_data[][][][][][]["lasers"][number of lasers (=32)]["dist" / "intensity"]
    # raw_data[][][]["data"]["stamp" / "type" / "value"]

    data_name = config.data_name[0]
    if file_exists(data_name, 'raw'):
        raw_data = load_pickle(data_name, 'raw')
    else:
        raw_data = readFile(data_name, progress=True)
        save_pickle(raw_data, data_name, 'raw')

    # calibration["lasers"][number of lasers (=64)]["dist_correction" / "dist_correction_x" / "dist_correction_y"/
    #     "focal_distance" / "focal_slope" / "horiz_offset_correction" / "laser_id" / "min_intensity" /
    #     "rot_correction" / "vert_correction" / "vert_offset_correction"]
    # calibration["num_lasers"] = 64

    calibration = load_yaml(config.yaml_name)

    img_data = raw2img(raw_data, calibration, progress=True)
    save_pickle(img_data, data_name, 'img')

    # print(img[0])
    print(len(img_data), len(img_data[0])) # 602, 64

    calibrated_data = raw2calibrated(raw_data, calibration, progress=True)
    save_pickle(calibrated_data, data_name, 'calibrated')
