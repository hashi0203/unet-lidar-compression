import os

def readfile(file_name, ext='.txt', path='../data'):
    f = open(os.path.join(path, file_name+ext), 'r')
    data = f.readlines()
    f.close()

    ret = {}
    ret[0] = {}
    stack = [ret, ret[0]]
    hidx = 1
    pidx = 0

    for d in data:
        d = d.strip()
        if d.endswith(':'):
            stack[-1][d[:-1]] = {}
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
            stack[-1][pidx] = {}
            stack.append(stack[-1][pidx])
            pidx += 1
        elif d.startswith("data"):
            l = d.split()
            v = [int(l[1][1:-1])] + [int(k[:-1]) for k in l[2:]]
            stack[-1][l[0][:-1]] = v
            stack = stack[:-1]
        elif d == "---":
            stack = stack[:-2]
            stack[-1][hidx] = {}
            stack.append(stack[-1][hidx])
            hidx += 1
        else:
            l = d.split()
            stack[-1][l[0][:-1]] = int(l[1])
    return ret


if __name__ == '__main__':
    data = readfile('parking-lot')
    print(data[0]["header"])
    print(len(data))
    print(len(data[0]["packets"]))

    # data[hidx]["header"]["seq" / "stamp" / "frame_id"]
    # data[hidx]["packets"][pidx]["stamp" / "data"]