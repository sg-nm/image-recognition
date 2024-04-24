# CLIPの実装

ここでは，ViT-S（画像エンコーダ）とTransformer（テキストエンコーダ）による[CLIP](https://arxiv.org/abs/2103.00020)の実装例を紹介します．

## Data

CLIPの原論文では，約4億の画像・キャプションペアが学習に用いられていますが，計算量削減のため，ここでは[Conceptual 12M](https://github.com/google-research-datasets/conceptual-12m)を使用します．
Conceptual 12Mのダウンロードは，[こちら](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc12m.md)を参照してください．
なお，最終的には数百GB-1TB程度のファイル容量となるため，注意してください．

ダウンロード後，本実装では，下記のようなディレクトリ構造になっていることを想定しています．

```
|- gcc12m_shards
    |-- gcc-conceptual-12m-000000.tar
    ...
    |-- gcc-conceptual-12m-001242.tar
```

Tokenizer構築に必要なファイル（bpe_simple_vocab_16e6.txt.gz）は[こちら](https://github.com/openai/CLIP/tree/main/clip)からダウンロードできます．


## CLIPの学習

下記のコマンドを実行することで，CLIPの学習が開始されます．デフォルトでは複数GPUによる学習が行われます．
```
bash run.sh
```

ミニバッチサイズ3072（GPU4枚）で20,000 iteration学習させたときの学習損失値の推移は以下の通りです．
<img src="https://github.com/sg-nm/image-recognition/assets/17783053/6206923a-a325-4f2e-a1a8-5ff817dee334" width="75%">

## CLIPの評価

以下を実行することで，学習済みCLIPのImageNet-1Kにおけるゼロショット画像分類精度を測ることができます．

```
bash run_eval.sh
```

学習済みモデルは，`./trained_model/epoch_latest.pt`として保存されていることを想定しています．
また，ImageNet-1Kは[こちら](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php)から各自でダウンロードする必要があります．

上記の学習コードで20,000　iteration学習させたCLIPのImageNet-1K（検証データ）に対するTop-1分類精度は23.4%，Top-5分類精度は47.3%です．
