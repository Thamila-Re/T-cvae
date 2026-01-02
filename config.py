"""
プロジェクト内のパラメータを管理するためのモジュール．
Updated to support both LSTM and Transformer architectures with proper defaults.

A) プログラムを書くときにやること．
  1) デフォルトパラメータを `Parameters` クラス内で定義する．
  2) コマンドライン引数を `common_args` 内で定義する．

B) パラメータを指定して実行するときにやること．
  1) `python config.py` とすると，デフォルトパラメータが `parameters.json` というファイルに書き出される．
  2) パラメータを指定する際は，Parametersクラスを書き換えるのではなく，jsonファイル内の値を書き換えて，
  `python -p parameters.json main.py`
  のようにjsonファイルを指定する．
"""

from asyncio import FastChildWatcher
from dataclasses import dataclass, field
from utils import dump_params
from argparse import ArgumentParser


@dataclass(frozen=True)
class Parameters:
    """
    プログラム全体を通して共通のパラメータを保持するクラス．
    ここにプロジェクト内で使うパラメータを一括管理する．
    """
    args: dict = field(default_factory=lambda: {})  # コマンドライン引数
    run_date: str = ''  # 実行時の時刻
    git_revision: str = ''  # 実行時のプログラムのGitのバージョン
    result_dir: str = '' # 出力ディレクトリのパス

    # Device info
    gpu_id: int = 0   # 使用するGPUの番号


    # Condition
    conditional_mode: bool = True   # 条件付き学習, 条件付き生成を行う場合はTrue
    condition_params: list = field(default_factory=lambda: ["Average path length"])  # preprocessでconditionとして与えるparameter
    condition_round: int = 1    # conditionの値の丸める桁数(GraphTune:1, FE:4)
    condition_size: int = 1     # condition size
    condition_values: dict = field(default_factory=lambda: {
        "Power-law exponent": [2.6, 3.0, 3.4],
        "Clustering coefficient":[0.1, 0.2, 0.3],
        "Average path length":[3.0, 4.0, 5.0],
        "Average degree":[3.0, 3.5, 4.0],
        "Edge density":[0.05, 0.075, 0.10],
        "Modularity":[0.4, 0.55, 0.7],
        "Diameter":[10, 20, 30],
        "Largest component size":[7.0, 14.0, 20.0],
    })


    # Dataset
    dataset_size: int = 5000 # 統計モデルの生成する個数
    number_of_nodes: int = 50 # 統計モデルのノード数

    ## BA model
    connection_new_edge: int = 3  # 新しいノードから既存のノードに張られるエッジ数

    ## WS model
    connection_neighbor_node: int = 3 # 最近傍k個のノードにエッジを張る
    rewiring_edge_probs: float = 0.3

    ## ER model
    edge_create_prob: float = 0.5 # エッジの生成確率


    # Preprocess
    split_size: dict = field(default_factory=lambda: {"train": 0.9, "valid": 0.1, "test": 0}) # データセットをtrain用とvalid用に分割する際のvalid用データの比率
    search: str = "dfs" # dfsかbfsかrfsか、2nd_rwか
    dfs_mode: str = "high_degree_first"  # ["high_degree_first", "normal", "low_degree_first"]から選択できる.
    rw_p: float = 1.00 
    rw_q: float = 1.00 # Bias parameter for 2nd order random walk.
    seq_augment_mode: str = "normal" # 2nd order random walkによるシーケンスデータの増加モード. ["normal", "minority"]から選択できる.
    seq_augment_ratio: int = 1 # 2nd order random walkによるシーケンスデータの増加倍率. seq_augment_modeがnormal以外の場合は最大の倍率となる. default to 1.
    ignore_label: int = 1500  # 5-tuples内のデータにおいて、無視するデータ
    normalize: bool = False   # conditionのデータを正規化する
    standardize: bool = False  # conditionのデータを標準化する
    
    ## base dataset path
    base_dataset_path: str = "./dataset/"
    
    ## reddit dataset
    reddit_path: str = "./data/reddit_threads/reddit_edges.json"
    
    ## twitter dataset
    twitter_path: str = "./data/edgelists_50/renum*"
    
    ## twitter augment dataset
    twitter_augment_dataset_path: str = 'augment_dataset/twitter_augment_30%_edge50_apl4_gene100_lm300.joblib'
    
    ## apl uniform dataset
    apl_uniform_dataset_path: str = './dataset_by_generater/graphs_for_fe_train.joblib'
    
    ## graphs(GraphTune generated + twitter) for training FE
    generated_and_twitter_path: str = './dataset_by_generater/relabeled_GraphTune_generated(3_4_5)+twitter.joblib'
    
    ## disentanglementを確かめるためのERランダムグラフモデルによって生成されたデータセット
    er_disentanglement_dataset_path: str = './data/er_graphs_10000_n_2_24_p_0.0_1.0/'


    # Data augmentation
    augment_mode: str = 'edge_random'   # エッジをランダムに張り替えて、データ増やす
    edge_random_p: float = 0.3          # エッジランダム化の対象のエッジ数の割合
    gene_graph_num: int  = 100          # 生成グラフ数
    loop_max: int        = 300          # 生成loopの最大試行回数
    apl_th: float        = 4.0          # APLの閾値(これ以上だと、追加してよい)


    # Graph features
    power_degree_border_line: float = 0.7 # 次数分布の冪指数を出すときに大多数のデータに引っ張られるせいで１次元プロットが正しい値から離れてしまうのでいくつかの値を除いて導出するための除く割合


    # Models
    ## GraphTune出力をFEへ入力する前にonehotへ変換するか(True: onehot, False: propability distribution)
    fe_preprocess_onehot: bool = True
    
    ## GraphTune出力をArgmaxをとり、確定されたグラフをnetworkxのグラフobjに変換し、conditionに対応する特徴量を計算するかどうか(True: 変換し、特徴量を計算する, False: 変換しないし、特徴量の計算もしない)
    fe_preprocess_convert_graph_obj: bool = True
    
    ## modelの種類
    model_set: list = field(default_factory=lambda: [
      "cvae", "cvae_with_fe", "fe", "cvae_for_2_tuples", "transformer_cvae", "hybrid_cvae"
    ])
    ### f_embを使うかどうかを制御するフラグ
    use_decoder_femb: bool = True 
    '''初期状態h0,c0の生成モード
      - 0 : 特徴量CでFillする (default)
      - 1 : 0でFillする
      - 2 : 潜在変数zを線形層f_h0c0に通す
      - 3 : 潜在変数z,特徴量Cを線形層f_h0c0に通す
    '''
    h0c0_mode: int = 0 
    '''SOSの生成モード
      - 0 : 潜在変数z, 特徴量Cを線形層f_sosに通す (default)
      - 1 : 0埋めに特徴量Cを連結する
      - 2 : 特徴量Cを線形層f_repに通す
      - 3 : 初期状態h_0からSOSを計算する
    '''
    sos_mode: int = 0
    # name of 5-tuple
    tuple_name: list = field(default_factory=lambda: ["tu", "tv", "lu", "lv", "le"])
    # flag to cat rep to sos and input
    cat_rep_to_sos_and_input: bool = False  # Changed to False to match your parameters.json
    ## GFCのパラメータ更新を停止するかどうか
    froze_gfc_params: bool = True

    ## FEのパラメータ更新を停止するかどうか
    froze_fe_params: bool = True
    
    ## dropout層に入力されたデータをdropさせる割合
    dropout: float = 0.5
    
    ## model hyper parameters
    model_params: dict = field(default_factory=lambda: {
      # Common parameters for all models
      'batch_size': 37, 
      'clip_th': 1.0, 
      'lr': 0.001, 
      'weight_decay': 0,
      'rep_size': 10, 
      'word_drop': 0,
      "alpha": 1, 
      "beta": 3, 
      "gamma": 300,
      "encoder_condition": True, 
      "decoder_h_c_condition": True, 
      "decoder_sequence_condition": True,
      
      # LSTM-specific parameters (for cvae)
      'emb_size': 227, 
      'en_hidden_size': 223, 
      'de_hidden_size': 250,
      
      # Feature extractor parameters (for cvae_with_fe)
      "fe_emb_size": 256, 
      "fe_hidden_size": 512, 
      "fe_rep_size": 1,
      "fe_lstm_num_layer": 1, 
      "fe_dropout": 0.294456,
      
      # Transformer-specific parameters (for transformer_cvae)
      "transformer_d_model": 256,  # Changed from 512 to 256 to match your parameters.json
      "transformer_encoder_layers": 1,  # Changed from 4 to 1 to match your parameters.json
      "transformer_decoder_layers": 1,  # Already 1
      "transformer_nhead": 1,  # Changed from 8 to 1 to match your parameters.json
      "transformer_dim_feedforward": 512,  # Changed from 2048 to 512 to match your parameters.json
      "transformer_dropout": 0.1,
      "sequence_model": "transformer"
    })


    # Train
    epochs: int = 10000               # エポック数
    model_save_point: int = 100       # modelをsaveするチェックポイント(エポック数)
    early_stopping_rounds: int = 300  # valid lossが指定回数更新されなければ、学習停止
    save_last_model: bool = False     # 最新の状態のmodelをsaveするかどうか


    # Tune
    opt_epoch: int = 30  # Optunaのトライアル数


    # Eval
    ## 現状、"power_degree", "cluster_coefficient", "distance", "size"
    eval_params: list = field(default_factory=lambda: [
      "Power-law exponent", "Clustering coefficient", "Average path length",
      "Average degree", "Edge density", "Modularity", "Diameter", "Largest component size"
    ])
    sampling_generation: bool = True        # 生成時に出力される分布からサンプリングするか最大値から取るか
    generate_edge_num: int = 357            # 生成するgraphのエッジの数
    number_of_generated_samples: int = 1000 # 生成されるグラフサンプルの数
    size_th: int = 0                        # 評価に用いるネットワークのサイズの閾値
    del_ratio: float = 0.3                  # ランダムに削除するエッジの割合を指定する

    # Sampling parameters
    # top-k サンプリングもしくはtop-pサンプリングを行わないなら'none'
    # top-k,top-pサンプリングの方法 'topk', 'topp', 'both' から選ぶ
    sampling_mode: str = 'none'
    # top-kサンプリングのk
    top_k: int = 10  # Changed from 20 to 10 to match your parameters.json
    # top-pサンプリングのp
    top_p: float = 0.9  # Changed from 1.0 to 0.9 to match your parameters.json
    # 温度付きsoftmaxの温度T
    softmax_temperature: float = 1.0
    # 温度Tのリスト
    temperature_list: list = field(default_factory=lambda: [i / 10 for i in range(10, 41)])  # Simplified


    # Visualize
    ## originalのデータセットから可視化用にサンプリングする数
    pickup_num: int = 300
    ## 非連結グラフを使用するかどうか
    use_unconnected_graphs: bool = False
    
    ## detail
    visualize_detail: dict = field(default_factory=lambda: {
      "twitter_pickup": [300, None, [None]],# 統計的手法の場合, [生成するグラフ数, データ次元数, [指定パラメータ1, 指定パラメータ2]]
      "reddit_pickup": [300, None, [None]] 
    })
    
    ## type
    visualize_types: dict = field(default_factory=lambda: {"Real_data":'bbb',\
        "AveragePathLength_3.0":'aaa',"AveragePathLength_0.4":'yyy',"Average_PathLength_0.5":'xxx'})
    
    ## dfs detail
    dfs_visualize_detail: dict = field(default_factory=lambda: {
      "twitter_pickup": [3, None, [None]],  # 統計的手法の場合, [生成するグラフ数, データ次元数, [指定パラメータ1, 指定パラメータ2]]
      "reddit_pickup": [3, None, [None]]
    })
    
    ## degree detail
    degree_dist_detail: dict = field(default_factory=lambda: {
      "twitter_pickup": [30, None, [None]]  # 統計的手法の場合, [生成するグラフ数, データ次元数, [指定パラメータ1, 指定パラメータ2]]
    })

    # Logging
    ## GraphTune+FEでdatasetのcondtition, GraphTune出力のcondition, FE出力をlogging
    logging_condition_by_graphtune_fe: bool = True

    ## datasetのグラフとそれに対応するGraphTune出力のArgmaxグラフを可視化
    logging_graph_vis_with_dataset_and_decoding: bool = True


    # Etc.
    ## グラフ特徴量の精度を計算する時、正解と分類される値の範囲([a, b] = {x-a <= x <= x+b | xはグラフ特徴量})
    acc_range: dict = field(default_factory=lambda: {
      "Power-law exponent": [0.1, 0.1], "Clustering coefficient": [0.01, 0.01],
      "Average path length": [0.05, 0.05], "Average degree": [0.1, 0.1],
      "Edge density": [0.005, 0.005], "Modularity": [0.02, 0.02], "Diameter": [0, 0],
      "Largest component size": [0, 0]
    })
    
    ## すべての乱数生成器のseed
    seed: int = 200

    # result_parent_dir: str = "result"
    input_graph_path: list = field(default_factory=lambda: [])  # 入力する生成グラフへのpath
    input_original_graph_num: int = 2000                        # 入力する ./data グラフの数
    
    # 複数回実行時の開始seedのlist
    seed_list: list = field(default_factory=lambda: [1050])


def common_args(parser: 'ArgumentParser'):
    """
    コマンドライン引数を定義する関数．

        Args:
            parser (:obj: ArgumentParser):
    """
    # Parameters
    parser.add_argument("-p", "--parameters", help="パラメータ設定ファイルのパスを指定．デフォルトはNone", type=str, default=None)
    parser.add_argument("--stdout_log_level", type=str, help="標準出力するログレベルを指定する", required=False, default="INFO")
    parser.add_argument("--fileout_log_level", type=str, help="ファイル出力するログレベルを指定する", required=False, default="DEBUG")

    # Specify Directory
    parser.add_argument("--result_dir", type=str, help="params.result_dirを上書きする. デフォルトはNone", default=None)
    
    # Dataset
    parser.add_argument("--dataset", type=str, help="使用するデータセット名", required=True)

    # Train
    ## preprocess
    parser.add_argument("--preprocess", help="train前にデータセットの前処理を行う", default=False, action='store_true')
    ### 前処理で呼び出す関数の種類を["dfs_5_tuples", "dfs_2_tuples"]のフラグから選ぶ
    parser.add_argument("--preprocess_type", help="前処理の種類", type=str, default="dfs_5_tuples")
    ## model
    parser.add_argument('--use_model', type=str, help="使用するモデル名", default=None)    # 学習対象のmodelをmodel_setから選択する
    parser.add_argument('--softmax', type=bool, help="CVAEのDecoder出力にsoftmaxをかけるかどうか", default=True)    # CVAEのDecoder出力にsoftmaxをかけるかどうか
    parser.add_argument('--fe_file', type=str, help="事前学習済みFEの重みへのPATH", default=None)  # 事前学習されたFEの重みへのPATHを指定
    ## チェックポイントから学習を始める場合
    parser.add_argument('--checkpoint_file', help="loadするモデルへのPATH", default=None)  # loadするモデルのPATH
    parser.add_argument('--init_epoch', type=str, default=None)  # epochの初期値

    # Tune
    ## 既存のDBを使用するなら、pathを記載. Noneなら、実行時の結果dir直下にdbファイルを配置する.
    parser.add_argument('--db_path', type=str, help="sqlite:///result/20220707_172742/fe_tune.db", default=None)

    # Eval
    parser.add_argument("--eval_model", help="評価対象のモデルへのPATH", type=str, default=None)
    parser.add_argument("--dfs_history", help="サンプリング過程を記録するグラフの個数", type=int, default=None)
    parser.add_argument("--edit_train_graph", help="Eval時にTrain Graphをランダムに編集したものを生成する", default=False, action='store_true')

    # Visualize
    parser.add_argument('--eval_graphs', help="生成されたグラフが格納されているディレクトリへのPATH", type=str, default=None)		# 生成されたグラフが保存されているディレクトリ
    parser.add_argument('--visualize_path', help="グラフを可視化した結果を格納するためのディレクトリへのPATH", type=str, default=None)		# グラフの可視化結果を保存するディレクトリ

    ## 以下はデフォルトでvisualizeするもの
    parser.add_argument("--out_csv", help="visualizeで見る必要がない場合引数として与える", default=True, action='store_false')
    parser.add_argument("--average_param", help="visualizeで見る必要がない場合引数として与える", default=True, action='store_false')
    parser.add_argument("--l2_norm", help="visualizeで見る必要がない場合引数として与える", default=True, action='store_false')
    parser.add_argument("--scatter_diagram", help="visualizeで見る必要がない場合引数として与える", default=True, action='store_false')
    parser.add_argument("--histogram", help="visualizeで見る必要がない場合引数として与える", default=True, action='store_false')
    parser.add_argument("--concat_scatter_diagram", help="visualizeで見る必要がない場合引数として与える", default=True, action='store_false')
    parser.add_argument("--concat_histogram", help="visualizeで見る必要がない場合引数として与える", default=True, action='store_false')
    parser.add_argument("--pair_plot", help="visualizeで見る必要がない場合引数として与える", default=True, action='store_false')
    parser.add_argument("--quality", help="visualizeで見る必要がない場合引数として与える", default=True, action='store_false')
    parser.add_argument("--unconnected_ratio", help="visualizeで見る必要がない場合引数として与える", default=True, action='store_false')
    # 以下は、計算時間が長いのでデフォルトではvisualizeしないもの
    parser.add_argument("--ged", help="visualizeで見る必要がある場合のみ引数として与える(約4時間かかる)", default=False, action='store_true')
    parser.add_argument("--skip_calc_ged", help="all_gedが計算できている場合に用いる", default=False, action='store_true')
    parser.add_argument("--history_vis", help="シーケンスへのサンプリング過程を可視化する場合に用いる 'png' 'gif' 'both'から選べる", type=str, default=None)

    # summarize_results
    parser.add_argument('--summarize_dir', help="複数回実行用やまとめるための結果を格納しているディレクトリへのPATH 末尾に/は付けない", type=str, default=None)
    parser.add_argument('--heatmap_dir', help="ヒートマップを作成するためのディレクトリへのPATH 末尾に/は付けない", type=str, default=None)

    # run_multiply
    parser.add_argument('--multiply_mode', help="複数回実行する際のモード 'eval' 'visualize' 'dimension'から選べる 複数選ぶ場合は'_'で繋げる", type=str, default=None)
    # Dimension reduction
    parser.add_argument("--reduction_target", help="次元削減の対象", type=str, default="latent_space")
    parser.add_argument("--reduction_method", help="次元削減の手法", type=str, default="tsne")
    parser.add_argument("--reduction_figure", help="次元削減後の図の形式", type=str, default="scatter")

    return parser


if __name__ == "__main__":
    dump_params(Parameters(), './', partial=True)  # デフォルトパラメータを出力