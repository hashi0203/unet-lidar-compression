import torch
# import torch.backends.cudnn as cudnn

import numpy as np
import matplotlib.pyplot as plt
import pillow_jpls
from PIL import Image
from io import BytesIO

import config
from utils import *
from data_loader import *
from model import LiDAR_UNet

import os
import argparse

import functools
print = functools.partial(print, flush=True)

parser = argparse.ArgumentParser(description='PyTorch LiDAR Point Cloud Compression Evaluation')
# --refresh, -r: refresh dataset files to update
parser.add_argument('--refresh', '-r', action='store_true', help='refresh dataset files')
# --progress, -p: use progress bar when preparing dataset
parser.add_argument('--progress', '-p', action='store_true', help='use progress bar')
args = parser.parse_args()


def quantize_2d(x, a=config.alpha):
    x *= 500
    r = [1 / (2 * a), 2 / (5 * a), 4 / (13 * a)]
    f = lambda x : x * r[0] if x <= 1000 else x * r[1] if x <= 5000 else x * r[2]

    return np.frompyfunc(f, 1, 1)(x).astype(np.int32)


print('==> Preparing data..')
data = loadLiDARData(refresh=args.refresh, progress=args.progress)
# 0 埋めしていないデータを読み出す

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Loading checkpoint..')
checkpoint = torch.load(config.CKPT_FILE)
model = LiDAR_UNet().to(device)

model.load_state_dict(checkpoint['model'])
model.eval()

print('==> Start compressing..')
if not os.path.isdir(config.GRAPH_PATH):
    os.mkdir(config.GRAPH_PATH)
HIST_NAME = os.path.join(config.GRAPH_PATH, 'hist-' + config.CKPT_FILE[-13:-4])

n = config.nbframe
with torch.no_grad():
    for i, d in enumerate(data):
        buf_ifs = BytesIO()
        buf_org = BytesIO()
        buf_out = BytesIO()

        d = np.array(d[:((len(d) - 1) // (n+1)) * (n+1) + 1])
        frames = d[:-1].reshape(-1, n+1, *d.shape[1:])
        I_frames = np.append(frames[:, 0], [d[-1]], axis=0)
        inputs = torch.tensor(list(zip(I_frames[:-1], I_frames[1:])))
        targets = torch.tensor(frames[:, 1:])

        # (C, H, W): (2, 64, 2088)
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.float)
        # inputs : (N,       2, 64, 2088)
        # targets: (N, nbframe, 64, 2088)

        outputs = torch.stack(tuple([model(inputs, t) for t in (np.arange(n) + 1) / (n + 1)])).permute(1, 0, 2, 3)
        # outputs: (N, nbframe, 64, 2088)

        plt.figure()
        plt.hist([inputs.cpu().reshape(-1), targets.cpu().reshape(-1), outputs.cpu().reshape(-1), (targets - outputs).cpu().reshape(-1)], label=['input', 'target', 'output', 'residual'], stacked=False, range=(-5, 5), density=True)
        plt.legend()
        plt.savefig(HIST_NAME + '-%d.png' % i)

        plt.figure()
        plt.hist([inputs.cpu().reshape(-1)], bins=50, range=(-3, 7), density=True)
        plt.title('input')
        plt.savefig(HIST_NAME + '-input-%d.png' % i)

        plt.figure()
        plt.hist([targets.cpu().reshape(-1)], bins=50, range=(-3, 7), density=True)
        plt.title('target')
        plt.savefig(HIST_NAME + '-target-%d.png' % i)

        plt.figure()
        plt.hist([outputs.cpu().reshape(-1)], bins=50, range=(-3, 7), density=True)
        plt.title('output')
        plt.savefig(HIST_NAME + '-output-%d.png' % i)

        plt.figure()
        plt.hist([(targets - outputs).cpu().reshape(-1)], bins=50, range=(-3, 7), density=True)
        plt.title('residual')
        plt.savefig(HIST_NAME + '-residual-%d.png' % i)

        for I in I_frames:
            Image.fromarray(quantize_2d(I), 'L').save(buf_ifs, "JPEG")

        img_org = targets.cpu().detach().numpy().copy()
        img_out = (targets - outputs).cpu().detach().numpy().copy()

        for j in range(img_org.shape[0]):
            for k in range(img_org.shape[1]):
                Image.fromarray(quantize_2d(img_org[j][k]), 'L').save(buf_org, "JPEG-LS")
                Image.fromarray(quantize_2d(img_out[j][k]), 'L').save(buf_out, "JPEG-LS")

        print('Movie %d: ' % i)
        print('\tmean: (input, residual) = (%.3f, %.3f)'
            % (torch.mean(torch.abs(inputs)).item(), torch.mean(torch.abs(targets - outputs)).item()))

        bytes_ifs = len(buf_ifs.getvalue())
        bytes_org = len(buf_org.getvalue())
        bytes_out = len(buf_out.getvalue())
        bytes_rate = (bytes_ifs + bytes_out) / (bytes_ifs + bytes_org)
        bytes_per_pixel = (bytes_ifs + bytes_out) / (I_frames.size + img_out.size)
        print('\tI frames bytes           : %d' % bytes_ifs)
        print('\tB frames bytes (original): %d' % bytes_org)
        print('\tB frames bytes (output)  : %d' % bytes_out)
        print('\toutput / original        : %.3f' % bytes_rate)
        print('\tbytes per pixel          : %.3f' % bytes_per_pixel)

print('==> Finish.')
