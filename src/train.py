import torch
import torch.optim as optim
# import torch.backends.cudnn as cudnn

from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt

import config
from utils import *
from data_loader import *
from model import LiDAR_UNet

import os
import argparse
import datetime

import functools
print = functools.partial(print, flush=True)

parser = argparse.ArgumentParser(description='PyTorch LiDAR Point Cloud Compression Training')
# --progress, -p: use progress bar when preparing dataset
parser.add_argument('--progress', '-p', action='store_true', help='use progress bar')
# --refresh, -r: refresh dataset files to update
parser.add_argument('--refresh', '-r', action='store_true', help='refresh dataset files')
# --summary, -s: show torchsummary to see the neural model structure
parser.add_argument('--summary', '-s', action='store_true', help='show torchsummary')
args = parser.parse_args()

print('==> Preparing train/test data..')
dataset = LiDARData(refresh=args.refresh, progress=args.progress)

trainset = datasetsLiDAR(dataset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=2)

testset = datasetsLiDAR(dataset, train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=10)
# testloader = torch.utils.data.DataLoader(testset, batch_size=10, num_workers=2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Building model..')
model = LiDAR_UNet().to(device)

if args.summary:
    summary(model, (2, 64, 2088))

# if device == 'cuda':
#     # Acceleration by using DataParallel
#     model = torch.nn.DataParallel(model)
#     cudnn.benchmark = True

criterion = torch.nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def train():
    model.train()
    train_loss = 0
    for inputs, targets in trainloader:
        optimizer.zero_grad()

        # (C, H, W): (2, 64, 2088)
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.float)
        # inputs : (N,       2, 64, 2088)
        # targets: (N, nbframe, 64, 2088)

        outputs = torch.stack(tuple([model(inputs, t) for t in (np.arange(config.nbframe) + 1) / (config.nbframe + 1)])).permute(1, 0, 2, 3)
        # outputs: (N, nbframe, 64, 2088)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(trainloader)


def test():
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.float)
            outputs = torch.stack(tuple([model(inputs, t) for t in (np.arange(config.nbframe) + 1) / (config.nbframe + 1)])).permute(1, 0, 2, 3)

            loss = criterion(outputs, targets)
            test_loss += loss.item()

    return test_loss / len(testloader)


t = datetime.datetime.now().strftime('%m%d-%H%M')
if not os.path.isdir(config.CKPT_PATH):
    os.mkdir(config.CKPT_PATH)
CKPT_FILE = os.path.join(config.CKPT_PATH, 'ckpt-%s.pth' % t)

epochs = np.arange(1, config.nepoch+1)
best_loss = None
train_loss = []
test_loss = []
for epoch in epochs:
    train_loss += [train()]
    test_loss += [test()]
    print('[Epoch: %3d] train loss: %.3f test loss: %.3f' % (epoch, train_loss[-1], test_loss[-1]))

    if (best_loss is None) or (test_loss[-1] < best_loss):
        print('Saving ckeckpoint..')
        state = {
            'model': model.state_dict(),
            'loss': test_loss[-1],
            'epoch': epoch
        }
        torch.save(state, CKPT_FILE)
        best_loss = test_loss[-1]

    scheduler.step()

if not os.path.isdir(config.GRAPH_PATH):
    os.mkdir(config.GRAPH_PATH)
GRAPH_FILE = os.path.join(config.GRAPH_PATH, 'loss-%s.png' % t)

# visualize loss change
plt.figure()

plt.plot(epochs, train_loss, label="train", color='tab:blue')
am = np.argmin(train_loss)
plt.plot(epochs[am], train_loss[am], color='tab:blue', marker='x')
plt.text(epochs[am], train_loss[am]-0.01, '%.3f' % train_loss[am], horizontalalignment="center", verticalalignment="top")

plt.plot(epochs, test_loss, label="test", color='tab:orange')
am = np.argmin(test_loss)
plt.plot(epochs[am], test_loss[am], color='tab:orange', marker='x')
plt.text(epochs[am], test_loss[am]+0.01, '%.3f' % test_loss[am], horizontalalignment="center", verticalalignment="bottom")

plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.legend()
plt.title('loss')
plt.savefig(GRAPH_FILE)
