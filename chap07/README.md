## SimCLRの実装

ここでは，[STL-10データセット](https://cs.stanford.edu/~acoates/stl10/)を用いたSimCLRの実装例を紹介します．

下記を実行することで，画像エンコーダ（ResNet-18）をSimCLRで最適化できます．

```
python main.py
```

下図左は学習初期のInfoNCE損失値の推移を表しています．下図右は同じく学習初期の正例サンプルの識別率の推移を示しています．

![simclr_log](https://github.com/sg-nm/image-recognition/assets/17783053/139c1abd-71a2-46df-b837-dca3d77f256d)
