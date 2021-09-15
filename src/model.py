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

        self.unet_computation = UNet(2, 4)
        self.unet_interpolation = UNet(8, 5)

    def forward(self, x0, t=0):
        # (C, H, W): (2, 64, 2088)
        # x0: (N, 2, 64, 2088)

        y0 = self.unet_computation(x0).permute(1, 0, 2, 3)
        # y0: (4, N, 64, 2088)

        I = x0.permute(1, 0, 2, 3)
        # I: (2, N, 64, 2088)
        F_hat = torch.cat((-(1-t)*t*y0[2:] + t*t*y0[:2], (1-t)*(1-t)*y0[2:] - t*(1-t)*y0[:2]))
        # F_hat: (4, N, 64, 2088)
        g_hat = self.warping(I, F_hat)
        # g_hat: (2, N, 64, 2088)
        x1 = torch.stack((I[0], g_hat[0], F_hat[0], F_hat[1], F_hat[3], F_hat[2], g_hat[1], I[1])).permute(1, 0, 2, 3)
        # x1: (N, 8, 64, 2088)

        y1 = self.unet_interpolation(x1).permute(1, 0, 2, 3)
        # y1: (5, N, 64, 2088)

        V = torch.stack((y1[0] * (1-t), (1 - y1[0]) * t))
        # V: (2, N, 64, 2088)
        Z = V[0] + V[1]
        # Z: (N, 64, 2088)
        g = self.warping(I, F_hat + torch.cat((y1[3:], y1[1:3])))
        # g: (2, N, 64, 2088)
        B_hat = (V[0] * g[0] + V[1] * g[1]) / Z
        # B_hat: (N, 64, 2088)
        return B_hat

    def warping(self, I, F_hat): # (2, N, 64, 2088)
        return torch.cat((self.grid_sample(I[0], F_hat[:2]), self.grid_sample(I[1], F_hat[2:])))

    def grid_sample(self, I, F_hat): # (1, N, 64, 2088)
        return F.grid_sample(I.view(-1, 1, *I.shape[1:]), F_hat.permute(1, 2, 3, 0),
                mode='bilinear', padding_mode='reflection', align_corners=True).permute(1, 0, 2, 3)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LiDAR_UNet().to(device)
    summary(model, (2, 64, 2088))
