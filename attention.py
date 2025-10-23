import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.l1 = nn.Linear(channels, channels // reduction)
        self.activate = nn.ReLU()
        self.l2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.l2(self.activate(self.l1(x)))
        return x

class ChannelAttentionBlock(nn.Module):
    def __init__(self, size, mlp_channels):
        super().__init__()

        self.maxpool = nn.MaxPool2d((size, size), stride=(size, size))
        self.avgpool = nn.AvgPool2d((size, size), stride=(size, size))
        self.mlp = MLP(mlp_channels)
        self.activate = nn.Sigmoid()
    
    def forward(self, x):
        x1 = self.mlp(self.maxpool(x))
        x2 = self.mlp(self.avgpool(x))
        channel_attention = self.activate(x1 + x2)
        return channel_attention

class SpatialAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.activate = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat)
        spatial_attention = self.activate(out)
        return spatial_attention

class CBAM(nn.Module):
    def __init__(self, size, mlp_channels):
        super().__init__()

        self.cab = ChannelAttentionBlock(size, mlp_channels)
        self.sab = SpatialAttentionBlock()
    
    def forward(self, x):
        cab_out = self.cab(x).unsqueeze(-1).unsqueeze(-1)
        x = x * cab_out
        sab_out = self.sab(x)
        x = x * sab_out
        return x
    
class DenseNet(nn.Module):
    def __init__(self, input, output, growth=32):
        super().__init__()

        self.conv1 = nn.Conv2d(input, growth, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(input + growth, growth, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(input + 2 * growth, growth, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(input + 3 * growth, growth, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(input + 4 * growth, output, kernel_size=3, stride=1, padding=1)
        self.activate = nn.LeakyReLU()

    def forward(self, x):
        x1 = self.activate(self.conv1(x))
        x2 = self.activate(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.activate(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.activate(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x = torch.cat((x, x1, x2, x3, x4), 1)
        x5 = self.conv5(x)
        return x5


class PrepNet(nn.Module):
    def __init__(self, size, in_channels, intermediate_channels, out_channels, mlp_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels + intermediate_channels, intermediate_channels, kernel_size=3, stride=1, padding=1)
        self.cbam = CBAM(size, mlp_channels)
        self.conv3 = nn.Conv2d(2 * intermediate_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activate = nn.LeakyReLU()
    
    def forward(self, x):
        x1 = self.activate(self.conv1(x))
        x2 = self.activate(self.conv2(torch.cat((x, x1), dim=1)))
        x = torch.cat((x1, x2), dim=1)
        x = self.cbam(x)
        out = self.activate(self.conv3(x))

        return out

class AttentionModule(nn.Module):
    def __init__(self, size, in_channels, out_channels, mlp_channels):
        super().__init__()

        self.ext1 = PrepNet(size, in_channels, 64, out_channels, mlp_channels)
        self.ext2 = PrepNet(size, in_channels, 64, out_channels, mlp_channels)
        self.dense = DenseNet(2 * out_channels, out_channels)

        self.combinator = nn.Sequential(
            CBAM(size, mlp_channels),
            nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding=1)
        )
    
    def forward(self, igi, cgi):
        x1 = self.ext1(igi)
        x2 = self.ext2(cgi)
        x = torch.cat((x1, x2), dim=1)
        x = self.dense(x)
        out = self.combinator(x)
        return out
