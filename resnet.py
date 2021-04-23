
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


"""## ネットワークの定義

研究でVision Transformerについて調べているときに気になったことを検証する
    Vision Transformerの理解が深まるし，コンペにも参加できる! (一石二鳥)

気になったこと
    Layer Normalizationはどれくらい効果があるの?そもそもLayer Normalizationとは?
    MLPやMulti-Head Attentionの前に正規化を行っている意味は?



元のネットワーク：
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

変更点(コメントを付けてるところ):
    batch norm -> layer norm(group norm (num_groups= 1))
        そもそもgroup normはResNetをつかって実験されてるものだった
        https://arxiv.org/abs/1803.08494


    normを畳み込みの前に持ってくる
        x : conv -> norm
        o : norm -> conv


結果：
    group normは効果があった(emnistでパラメータを変えなくても5ptくらい向上)
    normを畳み込みの前に持ってきたのは効果があったのかよくわからない．

"""



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        num_group = 1

        self.gn1 = nn.GroupNorm(num_group, in_planes) # here
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_group, planes)    # here
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.GroupNorm(num_group, in_planes),  # here
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.conv1(self.gn1(x))) # here
        out = self.conv2(self.gn2(out))       # here
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        num_group = 1
        self.gn1 = nn.GroupNorm(num_group, in_planes)  # here
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.gn2 = nn.GroupNorm(num_group, planes)     # here
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.gn3 = nn.GroupNorm(num_group, planes)     # here
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.GroupNorm(num_group, in_planes),    # here
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.conv1(self.gn1(x)))          # here
        out = F.relu(self.conv2(self.gn2(out)))        # here
        out = self.conv3(self.gn3(out))                # here
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, channels = 3):
        super(ResNet, self).__init__()
        self.in_planes = 64
        num_group = 1

        self.gn1 = nn.GroupNorm(num_group, channels)   # here
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(self.gn1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_class=10, channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_class, channels=channels)


def ResNet34(num_class=10, channels=3):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_class, channels=channels)


def ResNet50(num_class=10, channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_class, channels=channels)


def ResNet101(num_class=10, channels=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_class, channels=channels)


def ResNet152(num_class=10, channels=3):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_class, channels=channels)
