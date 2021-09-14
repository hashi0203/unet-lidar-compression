import config
from utils import *

def preprocess(packets, calibration, distance_resolution_m=0.002):
    npackets = len(packets)
    nblocks = len(packets[0]["data"]["blocks"])
    nlasers = len(packets[0]["data"]["blocks"][0]["lasers"])

    mu = config.mu # 100 フレームの平均
    theta = config.theta # 100 フレームの1シグマ区間に95%が含まれるような標準偏差

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


def raw2img(data, calibration):
    return [preprocess(d["packets"], calibration) for d in data]


if __name__ == '__main__':
    # data[number of sequences]["header"]["seq" / "stamp" / "frame_id"]
    # data[number of sequences]["packets"][number of packet data]["stamp" / "data"]
    # data[][][]["data"]["blocks"][number of blocks (=12)]["id" / "pos" / "lasers"]
    # data[][][][][][]["lasers"][number of lasers (=32)]["dist" / "intensity"]
    # data[][][]["data"]["stamp" / "type" / "value"]

    data_name = config.data_name[1]
    data = load_pickle(data_name)

    # calibration["lasers"][number of lasers (=64)]["dist_correction" / "dist_correction_x" / "dist_correction_y"/
    #     "focal_distance" / "focal_slope" / "horiz_offset_correction" / "laser_id" / "min_intensity" /
    #     "rot_correction" / "vert_correction" / "vert_offset_correction"]
    # calibration["num_lasers"] = 64

    calibration = load_yaml(config.yaml_name)

    img = raw2img(data, calibration)

    print(img[0])
    print(len(img), len(img[0])) # 602, 64
