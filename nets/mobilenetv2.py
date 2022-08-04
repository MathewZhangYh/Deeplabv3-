import math
import torch.nn as nn

def Conv3x3BNReLU(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

def Conv1x1BNReLU(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_channel = round(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.extend([#
                # 利用1x1卷积进行通道数的上升
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_channel),
                nn.ReLU6(inplace=True),])
        layers.extend([
                # 进行3x3的逐层卷积，进行跨特征点的特征提取
                nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=3, stride=stride, padding=1, groups=hidden_channel, bias=False),
                nn.BatchNorm2d(hidden_channel),
                nn.ReLU6(inplace=True),
                # 利用1x1卷积进行通道数的下降
                nn.Conv2d(in_channels=hidden_channel, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = (x + self.conv(x)) if self.use_res_connect else out
        return out

class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, alpha=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = int(32 * alpha)
        self.last_channel = int(1280 * alpha)

        interverted_residual_setting = [
            # t, c, n, s; expand_ratio, output_channel, 次数， stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],  #   2
            [6, 32, 3, 2],  #   4
            [6, 64, 4, 2],  #   7
            [6, 96, 3, 1],  #
            [6, 160, 3, 2], #   14
            [6, 320, 1, 1], #
        ]
        self.features = [Conv3x3BNReLU(3, input_channel, 2)]

        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * alpha)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        self.features.append(Conv1x1BNReLU(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )
        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x