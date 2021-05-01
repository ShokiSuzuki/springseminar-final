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
import csv
from torch.optim import lr_scheduler


parser = argparse.ArgumentParser()
#parser.add_argument('--epoch', help='define number of epoch', default=10, type=int)
#parser.add_argument('--batch_size', help='define batch size', default=4, type=int)
parser.add_argument('--layer', help='define layers', default=34, type=int)
args = parser.parse_args()

#num_epoch = args.epoch
#batch_size = args.batch_size
layers = args.layer


num_epoch = 15
batch_size = 256
num_class = 47
channels = 1

# 前処理を行う関数を複数定義
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomCrop(28),
    transforms.ToTensor(),                                   # データをテンソル型に変換
    transforms.Normalize(0.1736, 0.3317),
    transforms.RandomErasing()
])  # データの正規化（1つ目のタプル：各チャネルの平均， 2つ目のタプル：各チャネルの標準偏差)

test_transform = transforms.Compose([
    transforms.ToTensor(),                                   # データをテンソル型に変換
    transforms.Normalize(0.1736, 0.3317),
])  # データの正規化（1つ目のタプル：各チャネルの平均， 2つ目のタプル：各チャネルの標準偏差)


# 訓練データの読み込み
trainset = torchvision.datasets.EMNIST(root='./data', split='bymerge', train=True,
                                       download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

# テストデータの読み込み
testset = torchvision.datasets.EMNIST(root='./data', split='bymerge', train=False,
                                      download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2) # 注意！！　コンペに提出する場合は必ずshuffleをFalseに！

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
           'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't',)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if   layers == 18:
    net = resnet.ResNet18(num_class=num_class, channels=channels).to(device)
elif layers == 34:
    net = resnet.ResNet34(num_class=num_class, channels=channels).to(device)
elif layers == 50:
    net = resnet.ResNet50(num_class=num_class, channels=channels).to(device)
elif layers == 101:
    net = resnet.ResNet101(num_class=num_class, channels=channels).to(device)
else:
    net = resnet.ResNet152(num_class=num_class, channels=channels).to(device)
print(net)

#PATH = './emnist.pth'
#net.load_state_dict(torch.load(PATH))


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[3, 7], gamma=0.1)


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
        if i % 1000 == 999:    # 1000iterationごとに出力
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

    #scheduler.step()

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
        PATH = './emnist_best.pth'
        torch.save(net.state_dict(), PATH)
        best_score = score
        


print('Finished Training')


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
    print('Accuracy of %5s : %4d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


# create csv
with open('test_result_emnist.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "prediction"])


image_id = 10000   # csvファイルを結合するときのために10000から始める

with torch.no_grad():
    for data in testloader:
        # images, labels = data
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

        with open('test_result_emnist.csv', 'a') as f:
            writer = csv.writer(f)

            for i in range(len(predicted)):   # Dataloaderで設定したバッチサイズ
                writer.writerow([image_id, predicted[i].item()])
                image_id += 1
