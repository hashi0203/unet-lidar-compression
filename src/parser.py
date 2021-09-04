import os

def readfile(file_name, ext='.txt', path='../data'):
    f = open(os.path.join(path, file_name+ext), 'r')
    data = f.readlines()
    f.close()

    ret = [{}]
    stack = [ret, ret[0]]

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
            stack[-1][l[0][:-1]] = v
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
    # data[number of sequences]["header"]["seq" / "stamp" / "frame_id"]
    # data[number of sequences]["packets"][number of packet data]["stamp" / "data"]

    data = readfile('parking-lot')
    print(len(data)) # number of sequences: 602 (parking-lot), 600 (urban-road, residential-area)
    print(len(data[0]["packets"])) # number of packet data: 348
    print(data[0]["header"])
    print(data[1]["header"])
    print(data[2]["header"])
    print(data[-1]["header"])
