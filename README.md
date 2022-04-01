# EfficientNet

## 概要

このレポジトリは、AI病虫害画像診断システム　害虫判別器（EfficeintNet[^1]）の学習・推論プログラムを格納しています。

なお、本プログラムは、[tensorflow/tpu](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) の一部を改変したものです。[改変箇所の一覧](#改変箇所) もご参照ください。

## 動作環境

本プログラムの動作環境は下記のとおりです。

* Ubuntu 16.04 Xenial

学習・評価には、深層学習用の高速演算装置 [TPU](https://cloud.google.com/tpu/) を使うことができます。TPU を使用するには、
[Google Cloud Platform](https://cloud.google.com/) のアカウントが必要です。

## インストール手順

環境構築には、[Docker Engine](https://docs.docker.com/engine/) を使用します。GPU で推論する場合は、NVIDIA ドライバと [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) をインストールしてください。

動作環境（Docker コンテナ）は、下記のコマンドで作成できます。

```console
$ cd /path/to/cloned/repo   # このレポジトリのクローン先パス
$ docker build -t effnet ./
$ docker run -itdv /path/to/cloned/repo:/work/repo --name effnet --gpus all effnet bash
```

また、本プログラムは [Google Colaboratory](https://colab.research.google.com/) 上でも動作します。 Colaboratory での実行方法は、[付属のノートブック](efficientnet.ipynb)をご参照ください。

## データセットの準備

AI病虫害画像診断システムを構築するには、病害虫画像のデータセットが必要です。
構築に使用した画像とメタデータは、病害虫被害画像データベース (https://www.naro.affrc.go.jp/org/niaes/damage) で公開しています。

### データセットのディレクトリ構造

学習・評価に使う画像を、下記のようなディレクトリ構造で配置します。
```
my_dataset/
    train/
        aburamushi/
            100000_20201231123456_01.JPG
            200000_20201223234501_01.JPG
            ...
        azamiuma/
            ...
        ...
    validation/
        aburamushi/
            300000_20201231123456_01.JPG
            400000_20201223234501_01.JPG
            ...
        azamiuma/
            ...
        ...

{train, validation}: train は学習用、validation は評価用のサブセットです。
{aburamushi, azamiuma, ...}: 画像が属するクラスです。
```

### TFRecord 形式への変換
学習前に、データセットを TFRecord 形式に変換（シリアル化）する必要があります。 シリアル化には添付のスクリプト `efficientnet/imagenet_to_gcs.py` を使います。

次のコマンドで、TFRecord に変換します。

```console
$ datasetDir=/path/to/dataset    # /path/to/dataset を上記 dataset_root/ のパスに書き換えてください。

$ ## du: 各シャードが100MB程度になるように調整
$ docker exec -it effnet python /work/repo/tools/imagenet_to_gcs.py \
  --raw_data_dir "${datasetDir}" \
  --local_scratch_dir "${datasetDir}/tfrecord/" \
  --train_shards $(( $(du -Lad0 "${datasetDir}/train" | cut -f1) / 100000 + 1)) \
  --validation_shards $(( $(du -Lad0 "${datasetDir}/validation" | cut -f1) / 100000 + 1))
```

### アップロード（TPU 使用時のみ）

TPU で学習・評価する場合、データセットを Google Cloud Storage にアップロードする必要があります（Cloud Storage 以外の場所に置いたデータセットをTPUで読み込むことはできません）。

[`gsutil`](https://cloud.google.com/storage/docs/gsutil) コマンドを使って、TFRecord 形式のデータセットを Google Cloud Storage にアップロードします。

```console
$ gsutil -m cp -r /path/to/dataset/tfrecord gs://my-bucket/dataset/
```

### ShapeMask との併用

AI病虫害画像診断システム 害虫判別器の開発にあたっては、背景への過学習を抑制するために、

1. 葉や果実の形状を学習した ShapeMask モデルで背景を除去
1. 背景除去済みの画像を使って EfficientNet で学習・推論

という **2段階識別** の手法を採用しました。この手法を適用することで、非適用時と比べて top-1 正確度が向上することを確認しています。

EfficientNet 判別器を ShapeMask と併用する場合は、先に ShapeMask で背景除去処理を行い、背景のない画像を EfficientNet に学習させてください。

## 学習方法

学習は、下記のコマンドで実行することができます。

```console
$ docker exec -it effnet python /work/repo/main_v4rc5.py \
    --use_tpu="True" \
    --tpu="grpc://0.0.0.0:8470" \
    --data_dir="gs://path/to/dataset/" \
    --num_label_classes="10" \
    --model_dir="gs://path/to/model/" \
    --model_name="efficeintnet-b7" \
    --mode="train" \
    --input_image_size="1024" \
    --resize_method="None" \
    --train_batch_size="64" \
    --transfer_schedule="5,30" \
    --num_train_images="300000" \
    --iterations_per_loop="1000" \
    --warm_start_path="gs://path/to/previous/model/" \
    --augment_name="randaug" \
    --augment_subpolicy="zoom" \
    --base_learning_rate="0.016" \
    --mixup_alpha="0.2"
```

各コマンドライン引数の説明は、下記のとおりです。

* `--use_tpu`: TPU を使用するか（bool、False ならば CPU/GPU を使用）
* `--tpu`: TPU アドレス（`grpc://[TPU ノードのGRPCアドレス]:8470`、CPU/GPU 学習時は不要）
* `--data_dir`: データセットのあるディレクトリ
* `--num_label_classes`: ラベルのクラス数
* `--model_dir`: 学習済みモデルの保存先ディレクトリ
* `--model_name`: モデルの種類（`efficientnet-b1`、`efficientnet-b7` など）
* `--mode`: 動作モード (学習時は `train`)
* `--input_image_size`: 入力画像サイズ（指定した大きさにリサイズされる）
* `--resize_method`: リサイズ方式。`pad_and_resize` (パディング) または `None` (拡大・縮小)。
* `--train_batch_size`: 学習時バッチサイズ
* スケジュール（下記のいずれか1つ）
  * `--transfer_schedule`: 転移学習スケジュール
    `{head層学習エポック},{全レイヤ学習エポック}` を指定。
  * `--train_steps`: 学習ステップ数（全レイヤ学習のみ実施）
* `--num_train_images`: 学習画像の枚数
* `--iterations_per_loop`: チェックポイントの書き出し間隔（ステップ数で指定）
* `--warm_start_path`: 事前学習モデルのファイルパス
* `--augment_name`: 画像の水増し手法
  * `autoaugment`、`randaugment` または `None`。
* `--augment_subpolicy`: 画像水増し手法のサブポリシー
  * `augment_name == "autoaugment"` の場合のみ適用。`"v0"` (AutoAugment)、`"zoom"` (拡大・縮小) のいずれか。
* `--base_learning_rate`: 最大学習率
* `--mixup_alpha`: Mixup [^2] 水増しの係数（`0.0`=非適用）

学習が完了すると、`model_dir` に指定したディレクトリに学習済みモデルが保存されます。

## 評価方法

評価は、下記のコマンドで実行することができます。

```console
$ docker exec -it effnet python /path/to/efficientnet/main_v4rc5.py \
    --use_tpu="True" \
    --tpu="grpc://0.0.0.0:8470" \
    --data_dir="gs://path/to/dataset/" \
    --num_label_classes="10" \
    --model_dir="gs://path/to/model/" \
    --model_name="efficeintnet-b7" \
    --mode="eval" \
    --input_image_size="1024" \
    --eval_batch_size="64" \
    --num_eval_images="8192" \
    --steps_per_eval="1000" \
    --resize_method="None" \
    --eval_iterator="latest"
```

各コマンドライン引数の説明は、下記のとおりです。

* `--use_tpu`: TPU を使用するか（bool、False ならば CPU/GPU を使用）
* `--tpu`: TPU アドレス（`grpc://[TPU ノードのGRPCアドレス]:8470`、CPU/GPU 学習時は不要）
* `--data_dir`: データセットのあるディレクトリ
* `--num_label_classes`: ラベルのクラス数
* `--model_dir`: 学習済みモデルの保存先ディレクトリ
* `--mode`: 動作モード (評価時は `eval`)
* `--input_image_size`: 入力画像サイズ（指定した大きさにリサイズされる）
* `--eval_batch_size`: 評価時バッチサイズ
* `--num_eval_images`: 評価画像の枚数
* `--steps_per_eval`: 評価ステップの間隔
* `--resize_method`: リサイズ方式。`pad_and_resize` (パディング) または `None` (拡大・縮小)。
* `--eval_iterator`: チェックポイントの巡回方法
`all`: すべてのチェックポイント、`latest`: 最新のチェックポイントのみ、`[int]`: 指定したステップ以降のチェックポイント

評価が完了すると、評価結果（Top-1 精度、CSV 形式の混同行列など）が標準出力に書き出されます。

## 推論方法

未知の画像に対する推論は、下記のコマンドで実行できます。

```console
$ docker exec -it effnet python /path/to/efficientnet/eval_ckpt_main.py \
    --model_name="efficeintnet-b0" \
    --input_image_size=1024 \
    --ckpt_dir="/path/to/ckpt" \
    --example_img="/path/to/image.jpg" \
    --labels_map_file="/path/to/labels_map.txt"
```

各コマンドライン引数の説明は、下記のとおりです。

* `--model_name`: モデル名（`efficientnet-b{0..7}`）
* `--input_image_size`: 入力画像サイズ（指定した大きさにリサイズされる）
* `--ckpt_dir`: チェックポイントのあるディレクトリ
* `--example_img`: 推論対象の画像ファイルパス
* `--labels_map_file`: ラベルインデックスとラベル名の定義ファイル

## ライセンス

本プログラムは、Apache License 2.0 で提供されます。詳細は、[LICENSE](LICENSE) ファイルをご参照ください。

## 改変箇所

オリジナルのプログラム (https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) との主要な差異は、下記のとおりです。

* 他のモデルのパラメータを初期値として学習する機能を追加 (`--warm_start_path`, `--transfer_learning`)。
* 水増し手法の追加（90度ごとの回転、拡大・縮小など）。
* 評価時に、クラス別の再現率・適合率、混同行列を書き出す機能を追加。
* 評価時に、チェックポイントの巡回方法を指定する機能を追加（`--eval_iterator`）。
* `train_and_eval` モードを削除。
* 評価用のTFRecordデータも含めて学習する機能を追加（`--include_validation`）。
* 学習時にJSONファイルを指定することで、複数クラスを結合して学習する機能を追加（`--translate_classes`）。
* `eval_ckpt_main.py` に入力画像サイズの指定機能を追加（`--input_image_size`）。

[^1]: https://arxiv.org/abs/1905.11946
[^2]: https://arxiv.org/abs/1710.09412