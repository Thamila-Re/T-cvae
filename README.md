# Transformer-Based Conditional Graph Generation for Tunable Structural Features

## Requirement

- 仮想環境 : pyenv
  - Anacondaを使うとなぜかtensorboard関連のバージョンエラーが起きます。当面は、pyenvを使う必要があります。再現性が低くなるので、Anacondaでも環境を立てて比較検証をしようと考えています。(井脇)
- Python 3.7.6
- pytorch 1.9.0

## Installation

- 前処理後のデータセットが保存されるディレクトリを作成

```shell
mkdir dataset
```

- 結果出力用ディレクトリを作成

```shell
mkdir result
```

- 仮想環境の構築

```shell
python -m venv venv
```

- 各種モジュールのインストール

```shell
pip install --upgrade pip
pip install -r requirements.txt
```

- pytorch のインストール
  - 以下の URL を参照のこと.
    - <https://pytorch.org/get-started/previous-versions/>

## Dump Parameters

- デフォルトパラメータを json 形式で出力

```shell
python config.py
```

- 以降、`config.py`を編集せず、出力された`parameters.json`を適宜編集して使用する.

## Dataset

データセットを追加するときはファイル数が多いので、必ず圧縮してからプッシュする。
```sh
# 圧縮方法
$ tar -zcvf xxxxxx.tar.gz DATASET_DIR
# 解凍方法
$ tar -zxvf xxxxxx.tar.gz
```
## Training

- GraphTune の学習をバックグラウンドで実行(`train.py`を修正する必要あり).

```shell
nohup python -u train.py --preprocess --use_model cvae --dataset twitter -p parameters.json &
```

- GraphTune の学習をチェックポイントから再開する例(前処理された dataset は統一することに注意).　--init_epochからparams.epochまで学習が行われる。

```shell
nohup python -u train.py --use_model cvae --checkpoint_file result/{結果出力ファイル名}/train/weight_35000 --init_epoch 35001 -p parameters.json --dataset twitter &
```

- Feature Estimator(FE) の事前学習の例.

```shell
nohup python -u train_fe.py --preprocess --dataset twitter -p parameters.json &
```

- 事前学習済み FE を用いた CVAE の学習の例.

```shell
nohup python -u train_cvae_with_fe.py --preprocess --dataset twitter --fe_file result/{結果出力ファイル名}/train/valid_best_weight -p parameters.json &
```

- 2-tuples の DFS コードを扱う GraphTune の学習.

```shell
nohup python -u train.py --preprocess --preprocess_type dfs_2_tuples --use_model cvae_for_2_tuples --dataset twitter -p parameters.json &
```

## Alternative Training for GraphTune + FE

1. Choose one graph feature `F1` you want to learn.

2. Set the graph feature F1 you selected in GraphTune to be a conditon vector. You can edit `parameters.json` to do this.
```json
"conditional_mode": true, // whether to use a condition vector
"condition_params": [
  "Power-law exponent"    // F1
],
```

3. Train GraphTune only. You can use twitter dataset for example.
```shell
nohup python -u train.py --preprocess --use_model cvae --dataset twitter -p parameters.json &

Train with transformer encoder and transformer decoder:

nohup python -u train.py --preprocess --use_model transformer_cvae --dataset twitter -p parameters.json &

When need to train with the hybrid model:

nohup python -u train.py --preprocess --use_model hybrid_cvae --dataset twitter -p parameters.json &

For Transformer Encoder - LSTM decoder: change the sequence_model of parameter.json to "transformer"

For LSTM encoder-Transformer Decoder : change the sequence_model of parameter.json to "lstm"

all the other procedure similar to original graphtune.
```

4. Do conditional generation using trained GraphTune.
```shell
# example
python eval.py --use_model cvae --dataset twitter --eval_model result/{result_name}/train/valid_best_weight -p parameters.json
```

5. Concatenate all conditionally generated graphs and twitter dataset into one object `NEW_DATASET`.

6. Edit `config.py` and add the path to `NEW_DATASET` to the specified location.  Also, it is necessary to register the name of the `--dataset` argument when specifying `NEW_DATASET` as necessary.
```python
# example to add the path to NEW_DATASET

# ...
## graphs(GraphTune generated + twitter) for training FE
    generated_and_twitter_path: str = './dataset_by_generater/relabeled_GraphTune_generated(3_4_5)+twitter.joblib'
# ...
```

7. Train only FE with `NEW_DATASET`.
```shell
# example
nohup python -u train_fe.py --preprocess --dataset generated+twitter -p parameters.json &
```

8. Train GraphTune+FE.
  8.1. Using twitter dataset instead of `NEW_DATASET`
  8.2. Using untrained GraphTune
  8.3. Using trained FE
  8.4. Frozing FE parameters (FE parameters are not updated during training)
```shell
# example
nohup python -u train_cvae_with_fe.py --preprocess --dataset twitter --fe_file result/{result_name}/train/valid_best_weight -p parameters.json &
```

## Tuning

- FE の tuning の例.

```shell
nohup python -u tune_fe.py --preprocess --dataset twitter -p parameters.json &
```

- Optuna で使用した DB がある場合.

```sh
nohup python -u tune_fe.py --preprocess --db_path sqlite:///result/{結果出力ファイル名}/study.db --dataset twitter -p parameters.json &
```

## Evaluation

- 学習済みモデルを使用して条件付きグラフ生成をする例.

```shell
python eval.py --use_model cvae --dataset twitter --eval_model result/{結果出力ファイル名}/train/valid_best_weight -p result/{結果出力ファイル名}/parameters.json
```

- 学習済み FE の推定精度を確認する例.

```shell
python eval_fe.py --eval_model result/{結果出力ファイル名}/train/valid_best_weight --dataset twitter -p result/{結果出力ファイル名}/parameters.json
```

## Visualization

- 生成されたグラフを plot する例.

```shell
python visualize.py --eval_graphs result/{結果出力ファイル名}/eval/ --dataset twitter -p result/{結果出力ファイル名}/parameters.json
```

- Ubuntu の場合は, 計算グラフを`Graphviz`を使用して, 以下の手順で可視化できる.
  - `Graphviz`ライブラリの install

```shell
apt install -y --no-install-recommends graphviz graphviz-dev
```

- 計算グラフの可視化をするためのプログラムの例を以下に示す.
  - 結果は root ディレクトリの中に, デフォルトでは pdf 形式で保存される.

```python
from torchviz import make_dot
x1 = torch.tensor(1.0, requires_grad=True)
x2 = torch.tensor(2.0, requires_grad=True)
y = x1 * x2
dot = make_dot(y)
dot.render("example_file_name")
```

- model の出力から計算グラフを可視化する場合.

```python
from torchviz import make_dot
y = model(x)
# paramsを指定することにより, 計算グラフにmodelクラス内で定義された変数名が対応する箇所(node)に記載される.
image = make_dot(y, params=dict(model.named_parameters()))
image.format("png") # png形式で保存する
image.render("example_model_file_name")
```

- visualize.py には複数の関数が実装されているが、現状任意の関数を実行させる場合は visualize.py 下部の if **name** == "**main**":内を変更する必要がある。

- 複数のグラフをマトリクス状に可視化
  - 表示を調整するには現状bi.pyを直接編集するしかない．

```shell
bi.py
```

- GED(Graph Edit Distance)を計算し、CSVに出力する方法

```sh
python visualize.py --eval_graphs result/{結果出力ファイル名}/eval/ --dataset twitter -p result/{結果出力ファイル名}/parameters.json --ged
```

- GEDの計算には非常に時間がかかるため、`visualize.py`で計算できる他のパラメータについては計算しない方がいい。その場合、以下のようなオプションをつけることでその処理をスキップできる。指定できるオプションについては、`config.py`の`## Skip Visualize`以下を参照。また、`visualize.py`の更新に伴って`config.py`に適宜オプションを追加して欲しい。
- 以下は、特に処理の重たい`pair_plot`などをvisualizeしない場合のコマンド。
```sh
python visualize.py --eval_graphs result/{結果出力ファイル名}/eval/ --dataset twitter -p result/{結果出力ファイル名}/parameters.json --ged --scatter_diagram --hitogram --concat_scatter_diagram --concat_histogram --pair_plot
```

## Run MultiplyおよびSummarize Result
### Run Multiply
- train.pyによる学習済みの結果をディレクトリの同階層に保存し, 以下のコマンドを実行することで, 内部の全モデル (valid_best_weight) を用いた生成 (eval.py) および評価指標の算出 (visualize.py) を行う. グラフの生成数は実行時に -p で指定したparameters.json内の`generate_edge_num`で指定する. visualizeにおける時間がかかる処理や画像を保存する処理はスキップすることを推奨する.

```shell
nohup python run_multiply.py --dataset twitter -p parameters.json --use_model cvae --result_dir result/{結果出力ファイル名} --multiply_mode eval_visualize --histogram --scatter_diagram --concat_scatter_diagram --concat_histogram --pair_plot --summarize_dir result/{結果群が保存されているディレクトリ} &
```

- 上記のコマンドを実行することで, result/{結果群が保存されているディレクトリ}/{各結果ディレクトリ名}/ に以下のディレクトリが生成される. この際生成時のseedはparameters.json内の`seed_list`に記載されているseedが使用される. 非連結グラフを使用するかどうか (parameters.json内の`use_unconnected_graphs`) でディレクトリ名が変わる.
  - 非連結グラフを使用しない場合:
    - eval_seed{seed値}
    - visualize_seed{seed値}
  - 非連結グラフを使用する場合:
    - eval_seed{seed値}_unconnected
    - visualize_seed{seed値}_unconnected

### Summarize Result
- visualize_seed{seed値}* を用いて, 結果をまとめることができる. 以下のコマンドを実行することで, 生成されたグラフの統計量をまとめたcsvファイルが {--summarize_dir}/csv* 内に生成される. DFSやBFSの場合は一般的なテーブルだが, 2ndRWの結果をまとめた場合はピボットテーブルを保存する. 非連結グラフの結果をまとめる場合はparameters.json内の`use_unconnected_graphs`をTrueに設定する必要があり, そうでない場合はFalseに設定する必要がある. また, parameters.json内の`search`をまとめたい結果に対応させる必要がある.

```shell
python summarize_results.py --dataset twitter -p parameters.json --summarize_dir result/{結果群が保存されているディレクトリ}
```

- ヒートマップを作成する場合は, --summarize_dirとして2ndRWの結果群が保存されているディレクトリを指定する. 加えて, DFSとBFSの結果にsummarize_results.pyを実行し, --summarize_dirと同階層にそれらの結果群が存在する必要がある. つまり階層構造として以下のようになっている必要がある.
  ├── {--summarize_dir}
  │   ├── 2ndRW
  │   │   └── csv
  │   ├── DFS
  │   │   └── csv
  │   └── BFS
  │       └── csv

- 以下のコマンドを実行することで, 2ndRWの結果に対してヒートマップを作成することができる.

```shell
python summarize_results.py --dataset twitter -p parameters.json --summarize_dir result/{結果群が保存されているディレクトリ} --heatmap_dir result/{ヒートマップを保存する任意のディレクトリ}
```

- summarize_results.pyそのものにも複数回実行機能があり, 以下のような階層構造になっている場合に使用することができる.
  ├── {--summarize_dir} (データセットのサイズ等で分ける)
  │   ├── {指定した特徴量}
  │   │   ├── 2ndRW
  │   │   │   └── csv
  │   │   ├── DFS
  │   │   │   └── csv
  │   │   └── BFS
  │   │       └── csv
  │   ├── {指定した特徴量}
  │   │   ├── 2ndRW
  │   │   │   └── csv
  │   │   ├── DFS
  │   │   │   └── csv
  │   │   └── BFS
  │   │       └── csv
  ...

- 以下のコマンドを実行することで, 複数の結果を--heatmap_dir下にに保存することができる.

```shell
python summarize_results.py --dataset twitter --use_model cvae -p parameters.json --multiply_mode summarize --summarize_dir result/{結果群が保存されているディレクトリ} --heatmap_dir result/{ヒートマップを保存する任意のディレクトリ}
```

## Regeneration
- Average path length [3.0, ..., 10.0] のみ対応．基本的に training のコマンドライン引数と同様にしてよい
```shell
nohup python -u main_loop.py --preprocess --use_model cvae --dataset twitter [ -p parameters.json ] &
```

## Parameter Settings

- 指定できるパラメータは以下の通り. 詳細は`config.py`を参照.

```json
{
  "gpu_id": 0,
  "conditional_mode": true,
  "condition_params": ["Average path length"],
  "condition_round": 4,
  "condition_size": 1,
  "condition_values": {
    "Power-law exponent": [2.6, 3.0, 3.4],
    "Clustering coefficient": [0.1, 0.2, 0.3],
    "Average path length": [3.0, 4.0, 5.0],
    "Average degree": [3, 4, 5],
    "Edge density": [0.05, 0.075, 0.1],
    "Modularity": [0.5, 0.6, 0.7],
    "Diameter": [10, 20, 30],
    "Largest component size": [7.0, 14.0, 20.0]
  },
  "dataset_size": 2000,
  "number_of_nodes": 50,
  "connection_new_edge": 3,
  "connection_neighbor_node": 3,
  "rewiring_edge_probs": 0.3,
  "edge_create_prob": 0.5,
  "split_size": {
    "train": 0.9,
    "valid": 0.1,
    "test": 0
  },
  "dfs_mode": "high_degree_first",
  "ignore_label": 1500,
  "normalize": false,
  "standardize": false,
  "base_dataset_path": "./dataset/",
  "reddit_path": "./data/reddit_threads/reddit_edges.json",
  "twitter_path": "./data/edgelists_50/renum*",
  "twitter_augment_dataset_path": "augment_dataset/twitter_augment_30%_edge50_apl4_gene100_lm300.joblib",
  "apl_uniform_dataset_path": "./dataset_by_generater/graphs_for_fe_train.joblib",
  "generated_and_twitter_path": "./dataset_by_generater/relabeled_GraphTune_generated(3_4_5)+twitter.joblib",
  "augment_mode": "edge_random",
  "edge_random_p": 0.3,
  "gene_graph_num": 100,
  "loop_max": 300,
  "apl_th": 4.0,
  "search": "dfs",
  "power_degree_border_line": 0.7,
  "fe_preprocess_onehot": true,
  "fe_preprocess_convert_graph_obj": true,
  "model_set": ["cvae", "cvae_with_fe", "fe", "cvae_for_2_tuples"],
  "froze_fe_params": true,
  "dropout": 0.5,
  "model_params": {
    "batch_size": 37,
    "clip_th": 1.0,
    "lr": 0.001,
    "weight_decay": 0,
    "emb_size": 227,
    "en_hidden_size": 223,
    "rep_size": 10,
    "de_hidden_size": 250,
    "word_drop": 0,
    "fe_emb_size": 256,
    "fe_hidden_size": 512,
    "fe_rep_size": 1,
    "fe_lstm_num_layer": 1,
    "fe_dropout": 0.294456,
    "alpha": 1,
    "beta": 3,
    "gamma": 300,
    "encoder_condition": true,
    "decoder_h_c_condition": true,
    "decoder_sequence_condition": true
  },
  "epochs": 10000,
  "model_save_point": 100,
  "early_stopping_rounds": 300,
  "opt_epoch": 30,
  "eval_params": [
    "Power-law exponent",
    "Clustering coefficient",
    "Average path length",
    "Average degree",
    "Edge density",
    "Modularity",
    "Diameter",
    "Largest component size"
  ],
  "sampling_generation": true,
  "generate_edge_num": 357,
  "number_of_generated_samples": 300,
  "size_th": 0,
  "softmax_temperature": 1.0,
  "temperature_list": [
    1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
    2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
    4.0
  ],
  "visualize_detail": {
    "twitter_pickup": [300, null, [null]]
  },
  "visualize_types": {
    "Real_data": "bbb",
    "AveragePathLength_3.0": "aaa",
    "AveragePathLength_0.4": "yyy",
    "Average_PathLength_0.5": "xxx"
  },
  "dfs_visualize_detail": {
    "twitter_pickup": [3, null, [null]]
  },
  "degree_dist_detail": {
    "twitter_pickup": [30, null, [null]]
  },
  "logging_condition_by_graphtune_fe": true,
  "logging_graph_vis_with_dataset_and_decoding": true,
  "acc_range": {
    "Power-law exponent": [0.1, 0.1],
    "Clustering coefficient": [0.01, 0.01],
    "Average path length": [0.05, 0.05],
    "Average degree": [0.1, 0.1],
    "Edge density": [0.005, 0.005],
    "Modularity": [0.02, 0.02],
    "Diameter": [0, 0],
    "Largest component size": [0, 0]
  },
  "seed": 200,
  "seed_list": [1150, 1250, 1350]
}
```

## Directory Structure

- プロジェクトの構成

```shell
.
├── data                    # dataset
│   ├── Twitter
│   │   └── edgelists
│   ├── Twitter_2000
│   ├── csv
│   ├── edgelists_50        # すべて50 nodesのTwitterグラフデータセット. データ数は2,000.
│   └── reddit_threads
├── dataset                 # 前処理されたdataset
│   ├── train
│   └── valid
├── graph_process           # グラフ処理に関するモジュール
│  
├── models                  # 機械学習モデル
│  
├── tests                   # testファイルが格納されたdir
│  
├── result
    └── example
        ├── train           # train時のcheckpointとlossがminimumな時のモデルの重みを格納するdir
        ├── eval            # モデルによって生成されたグラフを格納するdir
        ├── visualize       # csvや可視化結果を格納するdir
        ├── parameters.json # 実行時のグローバル変数
        ├── example.db      # tune時に作成されるdb file
        └── example.log     # train, tune, evalなどのlog
```
