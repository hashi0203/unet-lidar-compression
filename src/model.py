import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.alpha = 0.1

        self.convs = []
        self.filter_sizes = [7, 5] + [3] * 10
        self.channel_sizes = [1, 32, 64, 128, 256, 512, 512, 512, 256, 128, 64, 32, 1]
        for i in range(len(self.filter_sizes)):
            ich = self.channel_sizes[i]
            och = self.channel_sizes[i+1]
            f = self.filter_sizes[i]
            pad = f // 2

            convs = [nn.Conv2d(ich, och, (f, f), padding=pad)]
            if i != len(self.filter_sizes) - 1:
                convs.append(nn.Conv2d(och, och, (f, f), padding=pad))

            self.convs.append(nn.ModuleList(convs))

        self.convs = nn.ModuleList(self.convs)

    def forward(self, x):
        shortcuts = []
        y = x.view(-1, 1, *x.shape[1:])
        for i in range(len(self.filter_sizes) // 2):
            if i != 0: y = F.max_pool2d(y, (2, 2))
            y = F.leaky_relu(self.convs[i][0](y), self.alpha)
            y = F.leaky_relu(self.convs[i][1](y) + y, self.alpha)
            shortcuts.append(y)

        shortcuts = shortcuts[:-1]

        for i in range(len(self.filter_sizes) // 2 - 1):
            l = i + len(self.filter_sizes) // 2
            y = F.interpolate(y, size=shortcuts[-i-1].shape[-2:], mode='bilinear', align_corners=True)
            y = F.leaky_relu(self.convs[l][0](y), self.alpha)
            y = F.leaky_relu(self.convs[l][1](y), self.alpha)
            y += shortcuts[-i-1]

        y = F.leaky_relu(self.convs[-1][0](y), self.alpha)

        return y

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet().to(device)
    summary(model, (64, 602))
