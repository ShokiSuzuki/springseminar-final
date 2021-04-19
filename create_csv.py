import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import resnet


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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
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



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = resnet.ResNet34().to(device)
print(net)


# create csv
import csv
with open('test_result.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "prediction"])

image_id = 0

with torch.no_grad():
    for data in testloader:
        # images, labels = data
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

        with open('test_result.csv', 'a') as f:
            writer = csv.writer(f)

            for i in range(len(predicted)):   # Dataloaderで設定したバッチサイズ
                writer.writerow([image_id, predicted[i].item()])
                image_id += 1

