from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.mobilenetv2 import MobileNetV2

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
