## MLPによる画像分類（MNIST）

本書内で紹介した2層MLPによる手書き数字（MNISTデータセット）の画像分類を行うコード例です．
モデルの最適化部分にはPyTorchを用いず，Numpyのみで実装しています．

![image_classification](https://github.com/sg-nm/image-recognition/assets/17783053/ed1c6d69-a598-4f58-9ace-0c657ca8e254)


下記を実行することで，MLPを学習できます．
```
python train_mnist.py
```

#### 学習の推移
```
# training log
Epoch [1/50], Loss: 4.325665746663806, Accuracy: 70.0642730496454%
Epoch [2/50], Loss: 2.105386336164403, Accuracy: 83.29787234042553%
Epoch [3/50], Loss: 1.6802360885934369, Accuracy: 85.68705673758865%
Epoch [4/50], Loss: 1.4313332902819353, Accuracy: 87.03180407801418%
Epoch [5/50], Loss: 1.2604553027311836, Accuracy: 87.98204787234043%
Epoch [6/50], Loss: 1.1331753786219174, Accuracy: 88.5549645390071%
Epoch [7/50], Loss: 1.0301088452768088, Accuracy: 89.00653812056737%
Epoch [8/50], Loss: 0.9365612582087478, Accuracy: 89.5595079787234%
Epoch [9/50], Loss: 0.8685705964418842, Accuracy: 89.7268395390071%
Epoch [10/50], Loss: 0.8070218459608335, Accuracy: 90.13685726950354%
...
```
