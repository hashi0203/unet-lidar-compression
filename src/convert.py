import pickle
import yaml

def calibrate_dist(packets, calibration, distance_resolution_m=0.002):
    npackets = len(packets)
    nblocks = len(packets[0]["data"]["blocks"])
    nlasers = len(packets[0]["data"]["blocks"][0]["lasers"])

    mu = 15 # 100 フレームの平均
    theta = 8 # 100 フレームの1シグマ区間に95%が含まれるような標準偏差

    img = [[packets[p]["data"]["blocks"][(2 * b) + (l // nlasers)]["lasers"][l % nlasers]["dist"] for b in range((nblocks // 2)) for p in range(npackets)] for l in range(nlasers * 2)]
    for i in range(nlasers * 2):
        for j in range(npackets * (nblocks // 2)):
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
    return [calibrate_dist(d["packets"], calibration) for d in data]


if __name__ == '__main__':
    # data[number of sequences]["header"]["seq" / "stamp" / "frame_id"]
    # data[number of sequences]["packets"][number of packet data]["stamp" / "data"]
    # data[][][]["data"]["blocks"][number of blocks (=12)]["id" / "pos" / "lasers"]
    # data[][][][][][]["lasers"][number of lasers (=32)]["dist" / "intensity"]
    # data[][][]["data"]["stamp" / "type" / "value"]

    with open('../data/parking-lot.bin', 'rb') as f:
        data = pickle.load(f)

    # calibration["lasers"][number of lasers (=64)]["dist_correction" / "dist_correction_x" / "dist_correction_y"/
    #     "focal_distance" / "focal_slope" / "horiz_offset_correction" / "laser_id" / "min_intensity" /
    #     "rot_correction" / "vert_correction" / "vert_offset_correction"]
    # calibration["num_lasers"] = 64

    with open('../data/64S2.yaml', 'r') as yml:
        calibration = yaml.load(yml, Loader=yaml.SafeLoader)

    img = raw2img(data, calibration)

    print(img[0])
