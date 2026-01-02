"""
生のデータセットを前処理し、modelの入力形式に変換するモジュール.
"""
# 標準
import copy
import glob
import random
import shutil
# Third-party
import joblib
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from graph_process import complex_networks, convert_to_sequence_code, graph_sampling
import utils
from config import Parameters
from logging import getLogger
from time import sleep

logger = getLogger(__name__)

def preprocess(params:Parameters, train_directory='./dataset/train/', valid_directory='./dataset/valid/', test_directory='./dataset/test/') -> None:
    """
    生データセットからグラフオブジェクトを作成し、それとそのサイズを保存する関数.

        Args:
            params (config.Parameters)  : グローバル変数のセット
            train_directory      (str)  : 前処理後のtrain用データセットが保存されるディレクトリ
                default: './dataset/train/'
            valid_directory      (str)  : 前処理後のvalidation用データセットが保存されるディレクトリ
                default: './dataset/valid/'
            test_directory       (str)  : 前処理後のtest用データセットが保存されるディレクトリ
                default: './dataset/test/'
    """
    logger.info("Start preprocess.")
    
    complex_network = complex_networks.ComplexNetworks(params)
    if params.args["dataset"] == 'disentangle_er':
        complex_network.create_disentanglement_er_dataset()
    else:
        complex_network.create_dataset(original_num=params.input_original_graph_num, input_graph_path=params.input_graph_path)
    
    train_dfs, train_time_set, train_node_set, train_max_length, train_label = sequence_code(params, "train")
    valid_dfs, valid_time_set, valid_node_set, valid_max_length, valid_label = sequence_code(params, "valid")
    if not params.split_size["test"] == 0:
        test_dfs, test_time_set, test_node_set, test_max_length, test_label = sequence_code(params, "test")
    else:
        # testがない場合は、trainのset{}を代入する
        test_time_set = train_time_set
        test_node_set = train_node_set
        test_max_length = 0
    
    # label standardization
    if params.standardize: # default : False
        if params.split_size["test"] == 0:
            label_data_list = [train_label, valid_label]
        else:
            label_data_list = [train_label, valid_label, test_label]
        # Standardization(train, valid, testそれぞれで平均・標準偏差を算出して標準化)
        for label_data in label_data_list:
            std_, mean_ = torch.std_mean(label_data, unbiased=True)
            for i, label in enumerate(label_data):
                label_data[i] = (label - mean_) / max(std_, 1e-7)

    # label normalization
    if params.normalize: # default : False
        if params.split_size["test"] == 0:
            label_data_list = [train_label, valid_label]
        else:
            label_data_list = [train_label, valid_label, test_label]
        # 最大値と最小値をtrainから探す
        train_max_val, train_min_val = -1, 10000
        for i, label in enumerate(train_label):
            if train_max_val < label.item():
                train_max_val = label.item()
            if train_min_val > label.item():
                train_min_val = label.item()
        # Normalization(trainに合わせて正規化)
        for label_data in label_data_list:
            for i, label in enumerate(label_data):
                label_data[i] = (label - train_min_val) / (train_max_val - train_min_val)

    # 登場するノードや次数の集合, 最大シーケンス長を計算する
    time_stamp_set = train_time_set | valid_time_set | test_time_set
    node_label_set = train_node_set | valid_node_set | test_node_set
    max_sequence_length = max(train_max_length, valid_max_length, test_max_length)
    conditional_label_length = params.condition_size
    print(f"max_sequence_length = {max_sequence_length}")

    node_label_set = set(range(50)) # 2回目以降の学習でnode_sizeを固定するために設定

    # ユニークなノード数, ユニークな次数, 条件付けラベルの長さを保存する
    joblib.dump([len(time_stamp_set)+1, len(node_label_set)+1, 2, conditional_label_length], f"./{params.result_dir}/dataset/param")

    # ノードID, ユニークな次数を辞書へ変換する
    time_dict = {time:index for index, time in enumerate(time_stamp_set)}
    node_dict = {node:index for index, node in enumerate(node_label_set)}
    del time_stamp_set, node_label_set

    # 深さ優先探索されたグラフをonehotなDFSコードへ変換し、保存する
    if params.args["preprocess_type"] == "dfs_5_tuples": # default
        # 5-tuplesのDFS codeを作成する
        get_onehot_and_list(train_dfs, time_dict,node_dict, max_sequence_length, train_label, train_directory, params.ignore_label)
        get_onehot_and_list(valid_dfs, time_dict,node_dict, max_sequence_length, valid_label, valid_directory, params.ignore_label)
        if not params.split_size["test"] == 0:
            get_onehot_and_list(test_dfs, time_dict, node_dict, max_sequence_length, test_label, test_directory, params.ignore_label)
    elif params.args["preprocess_type"] == "dfs_2_tuples":
        get_onehot_and_list_2_tuples(train_dfs, time_dict, node_dict, max_sequence_length, train_label, train_directory, params.ignore_label)
        get_onehot_and_list_2_tuples(valid_dfs, time_dict, node_dict, max_sequence_length, valid_label, valid_directory, params.ignore_label)
        if not params.split_size["test"] == 0:
            get_onehot_and_list_2_tuples(test_dfs, time_dict, node_dict, max_sequence_length, test_label, test_directory, params.ignore_label)
    else:
        print("不明なpreprocess_type.")
        exit()


def augment_dataset(params: 'config.Parameters', output_path: str):
    """Twitter dataset(node=50)のエッジをランダムに張り替えることで、新規のグラフを生成する

    Args:
        params (config.Parameters): グローバル変数のセット
        output_path (str): 生成されたグラフデータの保存先
                            e.g dataset/augment_data_50%_based_twitter_edges.joblib
    """
    # text形式のTwitter datasetをグラフobjへ変換
    text_files = glob.glob(params.twitter_path)
    text_files.sort()
    twitter_graphs = complex_networks.text2graph(text_files)

    if params.augment_mode == 'edge_random':
        # エッジランダム化
        extended_graphs = []
        for i, graph in tqdm(enumerate(twitter_graphs)):
            ## graph に対して、params.random_num個のedgeの張り替えられたグラフを作成
            generated_graph_cnt = 0
            graph_copy = graph.copy()
            ## APL 計算
            loop_cnt = 0    # 下記のwhile loopが回った回数

            while params.gene_graph_num > generated_graph_cnt:
                ## loop option
                if loop_cnt > params.loop_max:
                    break
                loop_cnt += 1
                ## 削除対象のエッジを選んで、削除
                remove_edge_num = int(len(graph_copy.edges()) * params.edge_random_p)
                remove_edge_indicies = random.sample(range(0, len(graph_copy.edges()), 1), k=remove_edge_num)
                graph_edge_list = [edge for edge in graph_copy.edges()]
                remove_edges = [edge for i, edge in enumerate(graph_edge_list) if i in remove_edge_indicies]
                graph_copy.remove_edges_from(remove_edges)
                ## エッジ削除した分だけ、ランダムにエッジ追加
                add_edge_lower = max(1, int(len(graph.edges())/2))
                add_edge_upper = 357
                add_edge_num_list = [x for x in range(add_edge_lower, add_edge_upper-1, 1)]
                # add_edge_num = random.sample(add_edge_num_list, k=1)[0]
                add_edge_num = remove_edge_num
                for _ in range(add_edge_num):
                    node_u = random.sample(graph_copy.nodes(), k=1)[0]
                    node_v_list = [node for node in graph_copy.nodes() if node != node_u]
                    node_v = random.sample(node_v_list, k=1)[0]
                    graph_copy.add_edge(node_u, node_v)

                # extended_graphs.append(graph_copy)
                # generated_graph_cnt += 1

                ## グラフが連結か調べる
                if nx.is_connected(graph_copy):
                    apl = nx.average_shortest_path_length(graph_copy)
                    ### グラフのAPLの条件がOKなら、追加
                    if params.apl_th <= apl:
                        extended_graphs.append(nx.freeze(graph_copy))
                        generated_graph_cnt += 1
                        graph_copy = nx.Graph(graph)
                    else:
                        del graph_copy
                        graph_copy = graph.copy()
                else:
                    del graph_copy
                    graph_copy = graph.copy()

        # Extended graphsの連結性を再度検証
        for i, graph in tqdm(enumerate(extended_graphs)):
            if not nx.is_connected(graph):
                print(i)
                # グラフの描画
                pos = nx.spring_layout(graph, seed=0)
                plt.figure(figsize=(10,10)) #グラフエリアのサイズ
                nx.draw_networkx(graph, pos) #グラフの描画(おまかせ)
                plt.savefig(f"graphs/graph_{i}.png") #グラフの描画
                plt.clf()
                plt.close()

        # 拡張されたグラフデータを保存
        print(f"len = {len(extended_graphs)}")
        joblib.dump(extended_graphs, output_path)
    else:
        print("config.Parameters.augment_mode が不明")


def concat_dataset(params: 'config.Parameters', input_path_list: list, output_path: str):
    """データセットを連結し、joblibで保存する関数

    Args:
        params (config.Parameters): グローバル変数のセット
        input_path_list (list): 入力pathのリスト
        output_path (str): 出力path
    """
    graphs = []
    for input_path in input_path_list:
        G_list = joblib.load(input_path)
        graphs = graphs + G_list
    print(f"len = {len(graphs)}")
    joblib.dump(graphs, output_path)


def sequence_code(params:Parameters, data_type:'str'):
    """
    labelが添付されたDFSコードを生成する関数

        Args:
            params (config.Parameters)  : グローバル変数のセット
            detail              (dict)  : 必要なデータセットに関する詳細

        Returns:
            (list) : dfs code
            (set)  : time stamp set
            (set)  : node label set
            (int)  : max sequence length
            (torch.Tensor) : label sets
    """
    complex_network = complex_networks.ComplexNetworks(params)
    datasets, labelsets = joblib.load( f"./{params.result_dir}/dataset/{str(params.args['dataset'])}_{data_type}")

    seq_code = list()
    time_stamp_set = set()
    nodes_label_set = set()
    max_sequence_length = 0
    augmented_labels = torch.Tensor()

    for index,graph in enumerate(datasets):
        if params.search == "2nd_rw":
            num_seq=int()
            if params.seq_augment_mode=="normal":
                if data_type == "train":
                    num_seq = params.seq_augment_ratio
                else:
                    num_seq = 1
            elif params.seq_augment_mode=="minority":
                pass # 少数データのみのaugmentationを実装する予定
            covert_graph = graph_sampling.GraphSampling(graph, node_order=params.dfs_mode, search_type=params.search, p=params.rw_p, q=params.rw_q)
            augmented_labels = torch.cat([augmented_labels, labelsets[index].expand(num_seq, -1).clone()], dim=0)
            for j in range(num_seq):
                tmp = covert_graph.get_seq_code()
                # 一旦tmpにdfscodeを出してからdfscodeにappend
                seq_code.append(tmp)
                # グラフの中の最大のシーケンス長を求める　+1はeosが最後に入る分
                if max_sequence_length < len(tmp)+1:
                    max_sequence_length = len(tmp)+1

                time_u = set(tmp[:, 0])
                time_v = set(tmp[:, 1])
                time = time_u | time_v
                time_stamp_set = time_stamp_set| time

                node_u = set(tmp[:,2])
                node_v = set(tmp[:,3])
                node = node_u | node_v
                nodes_label_set = nodes_label_set | node
        else:
            covert_graph = convert_to_sequence_code.ConvertToSequenceCode(graph, mode=params.dfs_mode, search_type=params.search)
            tmp = covert_graph.get_seq_code()
            # 一旦tmpにdfscodeを出してからdfscodeにappend
            seq_code.append(tmp)
            # グラフの中の最大のシーケンス長を求める　+1はeosが最後に入る分
            if max_sequence_length < len(tmp)+1:
                max_sequence_length = len(tmp)+1

            time_u = set(tmp[:, 0])
            time_v = set(tmp[:, 1])
            time = time_u | time_v
            time_stamp_set = time_stamp_set| time

            node_u = set(tmp[:,2])
            node_v = set(tmp[:,3])
            node = node_u | node_v
            nodes_label_set = nodes_label_set | node

    if(params.search == "2nd_rw"):
        labelsets = augmented_labels
    return seq_code, time_stamp_set, nodes_label_set,\
        max_sequence_length, labelsets

def get_onehot_and_list(dfs_code:list, time_dict:dict, node_dict:dict, max_sequence_length:int,
                        label_set:'torch.Tensor', directory:str, ignore_label:int):
    """指定されたサイズのone-hotへ変換する関数

    Args:
        dfs_code    (list): DFSコード
        time_dict   (dict): timestampのdict
        node_dict   (dict): node labelのdict
        max_sequence_length (int): 最大シーケンス長
        label_set  (torch.Tensor): conditionデータ
        directory    (str): onehotデータを保存するディレクトリ
        ignore_label (int): 5-tuples内で無視するラベル
    """
    time_end_num = len(time_dict.keys())
    node_end_num = len(node_dict.keys())
    dfs_code_onehot_list = []
    t_u_list = []
    t_v_list = []
    n_u_list = []
    n_v_list = []
    e_list = []
    for data in dfs_code:
        data = data.T
        # IDに振りなおす
        t_u = [time_dict[t] for t in data[0]]
        t_u.append(time_end_num)
        t_u = np.array(t_u)
        t_u_list.append(t_u)
        t_v = [time_dict[t] for t in data[1]]
        t_v.append(time_end_num)
        t_v = np.array(t_v)
        t_v_list.append(t_v)
        n_u = [node_dict[n] for n in data[2]]
        n_u.append(node_end_num)
        n_u = np.array(n_u)
        n_u_list.append(n_u)
        n_v = [node_dict[n] for n in data[3]]
        n_v.append(node_end_num)
        n_v = np.array(n_v)
        n_v_list.append(n_v)
        e = data[4]
        e = np.append(e,1)
        e_list.append(e)

        onehot_t_u = utils.convert2onehot(t_u,time_end_num+1)
        onehot_t_v = utils.convert2onehot(t_v,time_end_num+1)
        onehot_n_u = utils.convert2onehot(n_u,node_end_num+1)
        onehot_n_v = utils.convert2onehot(n_v,node_end_num+1)
        onehot_e = utils.convert2onehot(e,1+1)

        dfs_code_onehot_list.append(\
            np.concatenate([onehot_t_u,onehot_t_v,onehot_n_u,onehot_n_v,onehot_e],1))

    dfs_code_onehot_list = torch.Tensor(utils.padding(dfs_code_onehot_list,max_sequence_length,0))
    t_u_list = torch.LongTensor(utils.padding(t_u_list,max_sequence_length,ignore_label))
    t_v_list = torch.LongTensor(utils.padding(t_v_list,max_sequence_length,ignore_label))
    n_u_list = torch.LongTensor(utils.padding(n_u_list,max_sequence_length,ignore_label))
    n_v_list = torch.LongTensor(utils.padding(n_v_list,max_sequence_length,ignore_label))
    e_list = torch.LongTensor(utils.padding(e_list,max_sequence_length,ignore_label))

    joblib.dump(dfs_code_onehot_list, directory+'onehot')
    joblib.dump([t_u_list, t_v_list, n_u_list, n_v_list, e_list], directory+'label')
    joblib.dump(label_set, directory+'conditional')


def get_onehot_and_list_2_tuples(dfs_code:list, time_dict:dict, node_dict:dict, max_sequence_length:int,
                                label_set:'torch.Tensor', directory:str, ignore_label:int) -> None:
    """指定されたサイズのone-hotへ変換する関数(2-tuples)

    Args:
        dfs_code    (list): DFSコード
        time_dict   (dict): timestampのdict
        node_dict   (dict): node labelのdict
        max_sequence_length (int): 最大シーケンス長
        label_set  (torch.Tensor): conditionデータ
        directory    (str): onehotデータを保存するディレクトリ
        ignore_label (int): 2-tuples内で無視するラベル
    """
    time_end_num = len(time_dict.keys())
    dfs_code_onehot_list = []
    t_u_list = []
    t_v_list = []
    for data in dfs_code:
        data = data.T
        # IDに振りなおす
        t_u = [time_dict[t] for t in data[0]]
        t_u.append(time_end_num)
        t_u = np.array(t_u)
        t_u_list.append(t_u)
        t_v = [time_dict[t] for t in data[1]]
        t_v.append(time_end_num)
        t_v = np.array(t_v)
        t_v_list.append(t_v)

        onehot_t_u = utils.convert2onehot(t_u, time_end_num + 1)
        onehot_t_v = utils.convert2onehot(t_v, time_end_num + 1)

        dfs_code_onehot_list.append(np.concatenate([onehot_t_u, onehot_t_v], 1))

    dfs_code_onehot_list = torch.Tensor(utils.padding(dfs_code_onehot_list, max_sequence_length, 0))
    t_u_list = torch.LongTensor(utils.padding(t_u_list, max_sequence_length, ignore_label))
    t_v_list = torch.LongTensor(utils.padding(t_v_list, max_sequence_length, ignore_label))

    joblib.dump(dfs_code_onehot_list, directory + 'onehot')
    joblib.dump([t_u_list, t_v_list], directory + 'label')
    joblib.dump(label_set, directory + 'conditional')



if __name__ == "__main__":
    from config import common_args, Parameters
    from utils import setup_params
    import argparse
    parser = argparse.ArgumentParser()
    parser = common_args(parser)
    args = parser.parse_args()
    params = Parameters(**setup_params(vars(args), args.parameters))  # args，run_date，git_revisionなどを追加した辞書を取得
    
    preprocess(params=params,
                train_directory="tmp/ws_dataset/train/",
                valid_directory="tmp/ws_dataset/valid/",
                test_directory="tmp/ws_dataset/test/")
