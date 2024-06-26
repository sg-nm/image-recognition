## 「深層学習による画像認識の基礎」サポートページ

<img src="https://github.com/sg-nm/image-recognition/assets/17783053/41eab885-dc9f-49f7-8449-fc84010576fc" width="256px"><br><br>


書籍[「深層学習による画像認識の基礎」](https://www.ohmsha.co.jp/book/9784274231841/)（オーム社出版）のサポートページです。

本書で紹介したPython（[PyTorch](https://pytorch.org/get-started/locally/)）のソースコードおよび正誤表をまとめています。<br><br>

## 正誤表

本書の正誤情報は以下のページで公開しています。

[正誤表](https://github.com/sg-nm/image-recognition/wiki/Errata)

上記に掲載されていない誤植や間違いなどを見つけた方は，suganuma[at]vision.is.tohoku.ac.jpまでお知らせください。<br><br>

## コード内容

```
|- chap02
    |-- MLPによる手書き数字分類（MNIST）の実装
|- chap03
    |-- 畳込みフィルタの実装
|- chap04
    |-- 自己注意機構と層正規化の実装
|- chap05
    |-- Pix2Seqのモデル部分の実装
|- chap06
    |-- U-Netの実装
|- chap07
    |-- SimCLRの実装
|- chap08
    |-- CLIPの実装
```


## コードの主な動作環境

```
python 3.9.5
numpy==1.24.1
torch==2.2.2+cu118
torchvision==0.17.2+cu118
```

そのほかの必要なライブラリや依存関係については，`requirements.txt`を参照願います。

また，`chap07`および`chap08`の計算は非常に重いため，GPUを使用することを前提としています。

