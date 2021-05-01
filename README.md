# springseminar final

CIFAR100とEMNIST By Mergeの識別精度を競う．

## ネットワークの定義

### EMNIST By Merge用
ベース：ResNet ([ソース](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py))

#### 変更した箇所等
* 活性化関数をreluからgelu
* 畳み込みなどの順番(conv, norm, act, conv, norm, add, act -> norm, conv, act, norm, conv, add)
* 学習率をスケジューリング


### CIFAR-100用
ベース：Wide ResNet ([ソース](https://github.com/murarin/pytorch_models/blob/master/WideResNet.py))

#### 変更した箇所等
* 活性化関数をreluからgelu
* 学習率をスケジューリング


## まとめ
ネットワークの変更点はあまりないが、resnetについて調べていたら畳み込みなどの順番で精度が変わるとのことだったので、自分で実装してみた。
学習率のスケジューリングは、学習を早く終わらせたいために適当に値を決めて取り入れてみたら、学習率を小さくして学習していた時と同等の精度を早く出せるようになった。しかし、Wide ResNetはエポック数が少ないからか、本来の精度が出なかった。

