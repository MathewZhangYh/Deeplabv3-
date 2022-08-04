import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def ASPPConv(in_channels, out_channels, kernel_size, stride, padding, dilation):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True),
        nn.BatchNorm2d(out_channels, momentum=0.1),
        nn.ReLU(inplace=True),
    )

class ASPPPooling(nn.Sequential):
    def __init__(self, dim_in, dim_out):
        super(ASPPPooling, self).__init__(
            #nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=0.1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = torch.mean(x, 2, True)
        x = torch.mean(x, 3, True)
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, scale_factor=None, mode='bilinear', align_corners=True)

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
            # t, c, n, s;
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
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

class Deeplab_MobileNetV2(nn.Module):
    def __init__(self):
        super(Deeplab_MobileNetV2, self).__init__()
        model = MobileNetV2(n_class=1000)
        self.features = model.features[:-1]
        self.total_idx = len(self.features)  #18,0~17
        # downsample 16 # 最后一个block，14， 15, 16, 17
        for i in range(14, self.total_idx):
            self.features[i].apply(
                partial(self.down_sample, dilate=2)
            )

    def down_sample(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (1, 1) and m.kernel_size == (3, 3):
                m.dilation = (dilate, dilate)
                m.padding = (dilate, dilate)
            elif m.stride == (2, 2):
                m.stride = (1, 1)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x

#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ASPP, self).__init__()
        self.branch1 = ASPPConv(dim_in, dim_out, 1, 1, 0, 1)
        self.branch2 = ASPPConv(dim_in, dim_out, 3, 1, 6, 6)
        self.branch3 = ASPPConv(dim_in, dim_out, 3, 1, 12, 12)
        self.branch4 = ASPPConv(dim_in, dim_out, 3, 1, 18, 18)
        self.branch5 = ASPPPooling(dim_in,dim_out)

        self.conv_cat = nn.Sequential(
                nn.Conv2d(in_channels=dim_out*5, out_channels=dim_out, kernel_size=1, stride=1, padding=0,bias=True),
                nn.BatchNorm2d(dim_out, momentum=0.1),
                nn.ReLU(inplace=True),
        )

    def forward(self, x):
        #   一共五个分支
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        pool_conv = self.branch5(x)

        #   将五个分支的内容堆叠起来,然后1x1卷积整合特征。
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, pool_conv], dim=1)
        output = self.conv_cat(feature_cat)
        return output

class DeepLab(nn.Module):
    def __init__(self, num_classes):
        super(DeepLab, self).__init__()
        # 浅层特征[128,128,24];主干部分[30,30,320]
        self.backbone = Deeplab_MobileNetV2()
        in_channels = 320
        low_level_channels = 24
        #   ASPP特征提取模块
        self.aspp = ASPP(dim_in=in_channels, dim_out=256)

        #   浅层特征边
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(in_channels=low_level_channels, out_channels=48, kernel_size=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(in_channels=48+256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        # low_level_features: 浅层特征-进行卷积处理
        # x : 主干部分-利用ASPP结构进行加强特征提取
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)

        #   将加强特征边上采样与浅层特征堆叠后利用卷积进行特征提取
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
