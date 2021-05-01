# -*- coding: utf-8 -*-

# %matplotlib inline
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import resnet
import WideResNet
import argparse
import csv
from torch.optim import lr_scheduler


parser = argparse.ArgumentParser()
#parser.add_argument('--epoch', help='define number of epoch', default=10, type=int)
#parser.add_argument('--batch_size', help='define batch size', default=4, type=int)
parser.add_argument('--depth', help='define depth', default=16, type=int)
parser.add_argument('--width', help='define width', default=8, type=int)
args = parser.parse_args()

#num_epoch = args.epoch
#batch_size = args.batch_size
depth = args.depth
width = args.width


num_epoch = 25
batch_size = 256
num_class = 100
channels = 3

# 前処理を行う関数を複数定義
transform = transforms.Compose([
    #transforms.Resize(40),
    #transforms.RandomCrop(32),
    #transforms.RandomResizedCrop(32, scale=(0.7, 0.9)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),                                   # データをテンソル型に変換
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    transforms.RandomErasing()
])  # データの正規化（1つ目のタプル：各チャネルの平均， 2つ目のタプル：各チャネルの標準偏差)

test_transform = transforms.Compose([
    transforms.ToTensor(),                                   # データをテンソル型に変換
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])  # データの正規化（1つ目のタプル：各チャネルの平均， 2つ目のタプル：各チャネルの標準偏差)

# 訓練データの読み込み
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                     download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                      shuffle=True, num_workers=2)

# テストデータの読み込み
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                    download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
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



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net = WideResNet.WideResNet(depth=depth, num_classes=num_class, channels=channels, widen_factor=width, drop_rate=0.2).to(device)
print(net)



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

best_score = 0.0

for epoch in range(num_epoch):  # エポック数

    net.train()
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
        if i % 100 == 99:    # 100iterationごとに出力
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0


    scheduler.step()
    net.eval()
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

    score = 100 * correct / total
    print('Accuracy of the network on the test images: %.4f %%' % score)

    if score > best_score:
        # モデルの保存
        PATH = './cifar100.pth'
        torch.save(net.state_dict(), PATH)
        best_score = score

print('Finished Training')

PATH = './cifar100.pth'
net.load_state_dict(torch.load(PATH))


# クラスごとの精度
class_correct = list(0. for i in range(num_class))
class_total = list(0. for i in range(num_class))
with torch.no_grad():
    for data in testloader:
        # images, labels = data
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()

        for i, label in enumerate(labels, 0):    # Dataloaderで設定したバッチサイズ
            # label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1



# create csv
import csv
with open('test_result_cifar100.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "prediction"])

image_id = 0

with torch.no_grad():
    for data in testloader:
        # images, labels = data
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

        with open('test_result_cifar100.csv', 'a') as f:
            writer = csv.writer(f)

            for i in range(len(predicted)):   # Dataloaderで設定したバッチサイズ
                writer.writerow([image_id, predicted[i].item()])
                image_id += 1
