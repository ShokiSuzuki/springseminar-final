# springseminar final

CIFAR100とEMNIST By Mergeの識別精度を競う．

## ネットワークの定義

### EMNIST By Merge用
ベース：ResNet ([コピー元](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py))

#### 変更した箇所等
* 活性化関数をreluからgelu
* 畳み込みなどの順番(conv, norm, act, conv, norm, add, act -> )



### CIFAR-100用
