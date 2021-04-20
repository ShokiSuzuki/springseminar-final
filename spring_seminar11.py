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
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='define dataset', default='cifar100', type=str)
parser.add_argument('--epoch', help='define number of epoch', default=10, type=int)
parser.add_argument('--batch_size', help='define batch size', default=4, type=int)
args = parser.parse_args()

num_epoch = args.epoch
batch_size = args.batch_size


if args.dataset == 'cifar100':
    # 前処理を行う関数を複数定義
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),                                   # データをテンソル型に変換
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])  # データの正規化（1つ目のタプル：各チャネルの平均， 2つ目のタプル：各チャネルの標準偏差)

    # 訓練データの読み込み
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                           download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
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

    num_class = 100
    channels = 3

else:
    # 前処理を行う関数を複数定義
    transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(28),
        transforms.ToTensor(),                                   # データをテンソル型に変換
        transforms.Normalize(0.1736, 0.3317)
        ])  # データの正規化（1つ目のタプル：各チャネルの平均， 2つ目のタプル：各チャネルの標準偏差)

    # 訓練データの読み込み
    trainset = torchvision.datasets.EMNIST(root='./data', split='bymerge', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    # テストデータの読み込み
    testset = torchvision.datasets.EMNIST(root='./data', split='bymerge', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2) # 注意！！　コンペに提出する場合は必ずshuffleをFalseに！


    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
               'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
               'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't',)

    num_class = 47
    channels = 1


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = resnet.ResNet34(num_class=num_class, channels=channels).to(device)
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(num_epoch):  # エポック数

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
        if i % 2000 == 1999:    #  2000iterationごとに出力
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0


# モデルの保存
PATH = './{}.pth'.format(args.dataset)
torch.save(net.state_dict(), PATH)

print('Finished Training')

# PATH = './{}.pth'.format(args.dataset)
# net.load_state_dict(torch.load(PATH))

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


for i in range(num_class):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))



