## Pix2Seqの実装

ここでは，Pix2Seqのモデル部分の実装例を紹介しています．

![oveview_pix2seq](https://github.com/sg-nm/image-recognition/assets/17783053/6b92de5b-18e8-46e2-a542-0f7e481e182e)


なお，[原論文](https://arxiv.org/abs/2109.10852)では画像特徴はデコーダ内のクロス注意機構にキー・バリューベクトルとして入力されますが，ここでは簡略化のため，画像特徴をデコーダの入力系列（言語特徴）に結合する実装形式をとっています．

デコーダ内では，画像特徴間では通常の自己注意計算を行い，言語特徴に関してはマスク付きの自己注意計算（causal self-attention）を行っています．

Pix2Seqモデル全体
```
pix2seq.py
```

Transformerデコーダ
```
transformer.py
```
