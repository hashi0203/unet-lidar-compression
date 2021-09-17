import torch
# import torch.backends.cudnn as cudnn

import numpy as np
import matplotlib.pyplot as plt

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

        print('%d. mean: (input, residual) = (%.3f, %.3f)'
            % (i, torch.mean(torch.abs(inputs)).item(), torch.mean(torch.abs(targets - outputs)).item()))

print('==> Finish.')
