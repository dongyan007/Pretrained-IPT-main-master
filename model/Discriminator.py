import torch

from torch import nn

# 残差模块
from model import common
from option import args


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        self.net = nn.Sequential(
            # 第1个卷积层，卷积核大小为3×3，输入通道数为3，输出通道数为64
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            # 第2个卷积层，卷积核大小为3×3，输入通道数为64，输出通道数为64
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # 第3个卷积层，卷积核大小为3×3，输入通道数为64，输出通道数为128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # 第4个卷积层，卷积核大小为3×3，输入通道数为128，输出通道数为128
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # 第5个卷积层，卷积核大小为3×3，输入通道数为128，输出通道数为256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # 第6个卷积层，卷积核大小为3×3，输入通道数为256，输出通道数为256
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # 第7个卷积层，卷积核大小为3×3，输入通道数为256，输出通道数为512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # 第8个卷积层，卷积核大小为3×3，输入通道数为512，输出通道数为512
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # 全局池化层
            nn.AdaptiveAvgPool2d(1),
            # 两个全连接层，使用卷积实现
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 3, kernel_size=1))

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.net(x)
        x = self.add_mean(x)
        # batch_size = x.size(0)
        # return torch.sigmoid(self.net(x).view(batch_size))
        return x