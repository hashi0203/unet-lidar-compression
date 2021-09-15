import os
from tqdm import tqdm

import config
from utils import *

def concat_uint8(l):
    ret = 0
    for i in range(len(l)):
        ret |= l[i] << (i << 3)
    return ret


def parseData(data):
    assert len(data) == 1206

    ret = { "blocks" : [] }
    for block_head in range(0, 1200, 100):
        fring = data[block_head : block_head + 100]
        block = {}
        block["id"] = concat_uint8(fring[0:2])
        block["pos"] = concat_uint8(fring[2:4]) / 100.0
        block["lasers"] = []
        for laser_head in range(4, 100, 3):
            laser = {}
            laser["dist"] = concat_uint8(fring[laser_head : laser_head + 2])
            laser["intensity"] = fring[laser_head + 2]
            block["lasers"].append(laser)
        ret["blocks"].append(block)

    ret["stamp"] = concat_uint8(data[-6:-2])
    ret["type"] = data[-2]
    ret["value"] = data[-1]

    return ret


def readFile(data_name, ext='.txt', path='../data', progress=False):
    f = open(os.path.join(path, data_name+ext), 'r')
    data = f.readlines()
    f.close()

    ret = [{}]
    stack = [ret, ret[0]]

    if progress:
        data = tqdm(data)
        data.set_description("Parsing input file")

    for d in data:
        d = d.strip()
        if d.endswith(':'):
            stack[-1][d[:-1]] = [] if d.startswith("packets") else {}
            stack.append(stack[-1][d[:-1]])
        elif d.startswith("frame_id"):
            l = d.split()
            stack[-1][l[0][:-1]] = l[1][1:-1]
            stack = stack[:-1]
        elif d.startswith("nsecs"):
            l = d.split()
            stack[-1][l[0][:-1]] = int(l[1])
            stack = stack[:-1]
        elif d == '-':
            stack[-1].append({})
            stack.append(stack[-1][-1])
        elif d.startswith("data"):
            l = d.split()
            v = [int(l[1][1:-1])] + [int(k[:-1]) for k in l[2:]]
            stack[-1][l[0][:-1]] = parseData(v)
            stack = stack[:-1]
        elif d == "---":
            stack = stack[:-2]
            stack[-1].append({})
            stack.append(stack[-1][-1])
        else:
            l = d.split()
            stack[-1][l[0][:-1]] = int(l[1])

    return ret[:-1]


if __name__ == '__main__':
    # raw_data[number of sequences]["header"]["seq" / "stamp" / "frame_id"]
    # raw_data[number of sequences]["packets"][number of packet data]["stamp" / "data"]
    # raw_data[][][]["data"]["blocks"][number of blocks (=12)]["id" / "pos" / "lasers"]
    # raw_data[][][][][][]["lasers"][number of lasers (=32)]["dist" / "intensity"]
    # raw_data[][][]["data"]["stamp" / "type" / "value"]

    data_name = config.data_name[0]
    raw_data = readFile(data_name, progress=True)
    print(len(raw_data)) # number of sequences: 602 (parking-lot), 600 (urban-road, residential-area)
    print(len(raw_data[0]["packets"])) # number of packet data: 348
    print(raw_data[0]["header"])
    print(raw_data[1]["header"])
    print(raw_data[2]["header"])
    print(raw_data[-1]["header"])

    save_pickle(raw_data, data_name, 'raw')
