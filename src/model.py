import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class UNet(nn.Module):
    def __init__(self, ich, och):
        super(UNet, self).__init__()

        self.alpha = 0.1

        self.convs = []
        self.filter_sizes = [7, 5] + [3] * 10
        self.channel_sizes = [ich, 32, 64, 128, 256, 512, 512, 512, 256, 128, 64, 32, och]
        for i in range(len(self.filter_sizes)):
            ch1 = self.channel_sizes[i]
            ch2 = self.channel_sizes[i+1]
            f = self.filter_sizes[i]
            pad = f // 2

            convs = [nn.Conv2d(ch1, ch2, (f, f), padding=pad)]
            if i != len(self.filter_sizes) - 1:
                convs.append(nn.Conv2d(ch2, ch2, (f, f), padding=pad))

            self.convs.append(nn.ModuleList(convs))

        self.convs = nn.ModuleList(self.convs)

    def forward(self, x):
        shortcuts = []
        y = x
        for i in range(len(self.filter_sizes) // 2):
            if i != 0: y = F.avg_pool2d(y, (2, 2))
            y = F.leaky_relu(self.convs[i][0](y), self.alpha)
            y = F.leaky_relu(self.convs[i][1](y), self.alpha)
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

class LiDAR_UNet(nn.Module):
    def __init__(self):
        super(LiDAR_UNet, self).__init__()

        self.unet_computation = UNet(2, 2)
        self.unet_interpolation = UNet(6, 3)

    def forward(self, x, t=0):
        y = self.unet_computation(x).permute(1, 0, 2, 3)

        I = x
        F_hat = torch.stack((y[0] * t, y[1] * (1-t)))
        g_hat = self.warping(I, F_hat)
        y = torch.stack((I[0], F_hat[0], g_hat[0], g_hat[1], F_hat[1], I[1]))

        y = self.unet_interpolation(y.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)

        V = torch.stack((y[0] * (1-t), (1 - y[0]) * t))
        Z = V[0] + V[1]
        g = self.warping(I, F_hat + torch.stack((y[2], y[1])))
        B_hat = (V[0] * g[0] + V[1] * g[1]) / Z
        return B_hat

    def warping(self, I, F_hat):
        # todo
        return I


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LiDAR_UNet().to(device)
    summary(model, (2, 64, 602))
