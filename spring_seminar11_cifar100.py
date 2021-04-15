# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F

# 前処理を行う関数を複数定義
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),                                   # データをテンソル型に変換
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])  # データの正規化（1つ目のタプル：各チャネルの平均， 2つ目のタプル：各チャネルの標準偏差)

# 訓練データの読み込み
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

# テストデータの読み込み
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2) # 注意！！　コンペに提出する場合は必ずshuffleをFalseに！


classes = ("beaver", "dolphin", "otter", "seal", "whale",
  "aquarium fish", "flatfish", "ray", "shark", "trout",
  "orchids", "poppies", "roses", "sunflowers", "tulips",
  "bottles", "bowls", "cans", "cups", "plates",
  "apples", "mushrooms", "oranges", "pears", "sweet peppers",
  "clock", "computer keyboard", "lamp", "telephone", "television",
  "bed", "chair", "couch", "table", "wardrobe",
  "bee", "beetle", "butterfly", "caterpillar", "cockroach",
  "bear", "leopard", "lion", "tiger", "wolf",
  "bridge", "castle", "house", "road", "skyscraper",
  "cloud", "forest", "mountain", "plain", "sea",
  "camel", "cattle", "chimpanzee", "elephant", "kangaroo",
  "fox", "porcupine", "possum", "raccoon", "skunk",
  "crab", "lobster", "snail", "spider", "worm",
  "baby", "boy", "girl", "man", "woman",
  "crocodile", "dinosaur", "lizard", "snake", "turtle",
  "hamster", "mouse", "rabbit", "shrew", "squirrel",
  "maple", "oak", "palm", "pine", "willow",
  "bicycle", "bus", "motorcycle", "pickup truck", "train",
  "lawn-mower", "rocket", "streetcar", "tank", "tractor")

"""## ネットワークの定義"""


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = ResNet18().to(device)
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(20):  # エポック数

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 入力データの取得. 変数dataはリスト[inputs, labels]
        # inputs, labels = data  # cpuの場合はこっちでも可

        inputs, labels = data[0].to(device), data[1].to(device)

        # 勾配を0に初期化
        optimizer.zero_grad()

        # 順伝播、逆伝播、パラメータ更新
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # lossの出力
        running_loss += loss.item()
        if i % 2000 == 1999:    # 2000iterationごとに出力
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

# モデルの保存
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        # images, labels = data
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
