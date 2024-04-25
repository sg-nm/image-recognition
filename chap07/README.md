## SimCLRの実装

ここでは，[STL-10データセット](https://cs.stanford.edu/~acoates/stl10/)を用いた[SimCLR](https://arxiv.org/abs/2002.05709)の実装例を紹介します．

<img src="https://github.com/sg-nm/image-recognition/assets/17783053/cfded5b9-9738-41f4-82ae-bf01904e3b1e" width="50%"><br><br>

下記を実行することで，画像エンコーダ（ResNet-18）をSimCLRで最適化できます．

```
python main.py
```

<br>下図左は学習初期のInfoNCE損失値の推移を表しています．下図右は同じく学習初期の正例サンプルの識別率の推移を示しています．<br><br>

![simclr_log](https://github.com/sg-nm/image-recognition/assets/17783053/139c1abd-71a2-46df-b837-dca3d77f256d)
