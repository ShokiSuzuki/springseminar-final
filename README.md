# springseminar final

CIFAR100とEMNIST by mergeの識別精度を競う．

## ネットワークの定義

研究でVision Transformer(ViT)について調べているときに気になったことを検証する．
    Vision Transformerの気持ちが理解できるし，コンペにも参加できる! (一石二鳥)
ViTはResidual接続がされてるので，ResNetをベースとする．
元のネットワーク：
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

### 気になったこと
    Layer Normalizationはどれくらい効果があるの?そもそもLayer Normalizationとは?
    MLPやMulti-Head Attentionの前に正規化を行っている意味は?


### 変更点(resnet.pyでコメントを付けてるところ):
    batch norm -> layer norm(group norm (num_groups= 1))
        そもそもgroup normはResNetをつかって実験されてるものだった
        https://arxiv.org/abs/1803.08494


    normを畳み込みの前に持ってくる
        x : conv -> norm
        o : norm -> conv
