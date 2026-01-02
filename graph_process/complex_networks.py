"""
グラフに関する処理をまとめたモジュール.

主な機能は以下の通り.
・生のデータセットを読み込み、グラフオブジェクトに変換して保存する
・グラフオブジェクトからグラフ特徴量を算出し、csvファイルに書き出す
"""

import glob
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import torch
import os
import sys
from tqdm import tqdm
import random
import argparse
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from graph_process import graph_statistic
from graph_process.graph_utils import text2graph, sort_nodes_by_degree
from config import Parameters, common_args
import utils

from bi import pair_plot


class ComplexNetworks:
    """
    複雑ネットワークモデルを使用してデータセットを作成するクラス.
    """

    def __init__(self, params:'config.Parameters'):
        self.split_size = params.split_size
        self.condition_params = params.condition_params
        self.condition_round = params.condition_round
        self.base_dataset_path = params.base_dataset_path
        # Twitter
        self.twitter_path = params.twitter_path
        
        # Twitter augment
        self.twitter_augment_dataset_path = params.twitter_augment_dataset_path
        
        # apl uniform
        self.apl_uniform_dataset_path   = params.apl_uniform_dataset_path
        
        # GraphTune generated + twitter
        self.generated_and_twitter_path     = params.generated_and_twitter_path

        # 統計モデル関連の変数
        self.dataset_size = params.dataset_size

        self.number_of_node = params.number_of_nodes

        ## BAモデル関連の変数
        self.connection_new_edge = params.connection_new_edge
        
        ## WSモデル関連の変数
        self.connection_neighbor_node = params.connection_neighbor_node

        ## ERモデル関連の変数
        self.edge_create_prob = params.edge_create_prob
        
        ## Disentanglementを確かめるためのERモデル関連の辺数
        self.er_disentanglement_dataset_path = params.er_disentanglement_dataset_path

        self.run_date = params.run_date
        self.result_dir = params.result_dir
        self.eval_params = params.eval_params
        self.seed = params.seed

        self.args = params.args

    def create_seq_conditional_dataset(self, data_type):
        """
        条件付きシーケンスデータセットを作成する関数.

            Args:
                data_type (str) : データセットの種類(train, valid, test)
                
            Returns:
                (nx.graph) : グラフオブジェクト形式のデータセット
                (torch.tensor) : condition情報をもとに計算されたlabel
        """
        datasets = joblib.load(f"{self.base_dataset_path}{self.args['dataset']}_{data_type}")
        dataset, label = datasets[0], datasets[1]
        return dataset, label

    def create_dataset(self, original_num: int = None, input_graph_path: list = []):
        """
        テキストファイルからグラフオブジェクトを作成し、trainとvalidに分割して保存する関数.
        また、グラフ統計量を計算し、labelとして保存する.
        """

        # グラフ統計量を算出するためのインスタンス
        st = graph_statistic.GraphStatistic()

        if self.args["dataset"] == "twitter": # default
            text_files = glob.glob(self.twitter_path)
            text_files = sorted(text_files)
            graph_data = text2graph(text_files)
        elif self.args["dataset"] == "twitter_aug":
            with open(self.twitter_augment_dataset_path, 'rb') as f:
                graph_data = joblib.load(f)
        elif self.args["dataset"] == "apl_uniform":
            with open(self.apl_uniform_dataset_path, 'rb') as f:
                graph_data = joblib.load(f)
        elif self.args["dataset"] == "generated+twitter":
            with open(self.generated_and_twitter_path, 'rb') as f:
                graph_data = joblib.load(f)
        elif self.args["dataset"] == "BA":
            graph_data = []
            for _ in range(self.dataset_size):
                graph_data.append(nx.barabasi_albert_graph(
                    n=self.number_of_node,
                    m=self.connection_neighbor_node
                ))
            save_dir = f"./{self.result_dir}/dataset_detail/"
            utils.make_dir([save_dir])
            st.graph2csv(graph_data, save_dir, "BA_params", self.eval_params)
            pair_plot(self.eval_params, [save_dir+"BA_params.csv"], save_dir)
        elif self.args["dataset"] == "WS":
            graph_data = []
            while len(graph_data) != self.dataset_size:
                l = [0.1,0.2,0.3,0.4,0.5,0.6]

                graph = nx.watts_strogatz_graph(
                    n=self.number_of_node,
                    k=self.connection_neighbor_node,
                    p=random.choice(l)
                )
                if nx.is_connected(graph):
                    graph_data.append(graph)

            save_dir = f"./{self.result_dir}/dataset_detail/"
            utils.make_dir([save_dir])
            st.graph2csv(graph_data, save_dir, "WS_params", self.eval_params)
            pair_plot(self.eval_params, [save_dir+"WS_params.csv"], save_dir)
        elif self.args["dataset"] == "ER":
            graph_data = []
            for _ in range(self.dataset_size):
                graph_data.append(nx.erdos_renyi_graph(
                    n=self.number_of_node,
                    p=self.edge_create_prob
                ))
            save_dir = f"./{self.result_dir}/dataset_detail/"
            utils.make_dir([save_dir])
            st.graph2csv(graph_data, save_dir, "ER_params", self.eval_params)
            pair_plot(self.eval_params, [save_dir+"ER_params.csv"], save_dir)
        elif self.args["dataset"] == "reddit":  # twitter のデータセットに合わせ, 2000個のグラフをランダムに抽出する
            graph_data = [] 
            text_files = open(self.reddit_path,'r')
            text_files = json.load(text_files)
            for i in range(len(text_files)):
                G = nx.Graph()
                G.add_edges_from(text_files[str(i)])
                graph_data.append(G)
            graph_data = random.sample(graph_data, k=2000)
            
        else:
            raise Exception("不明なargs['dataset_type'].")
        
        #if original_num != None:
        #    graph_data = graph_data[:original_num]
        #for path in input_graph_path:
        #    input_graphs = joblib.load(path)
        #    for idx in range(len(input_graphs)):
        #        # 生成グラフのノードを次数でソートしてから学習データに連結する
        #        input_graphs[idx] = sort_nodes_by_degree(input_graphs[idx])
        #    graph_data.extend(input_graphs)

        # train dataとvalid dataのグラフ統計量の入れ物
        train_labels = torch.Tensor()
        valid_labels = torch.Tensor()

        # datasetをtrain, valid, testに分割する
        if self.split_size["test"] == 0:
            # when you do not need test data
            train_data, valid_data = train_test_split(graph_data, test_size=self.split_size["valid"], train_size=self.split_size["train"], random_state=0,
                                                        shuffle=True)
        else:
            # when you need test data
            train_data, valid_test_data = train_test_split(graph_data,
                                                            test_size=self.split_size["valid"]+self.split_size["test"],
                                                            train_size=self.split_size["train"],
                                                            random_state=0, shuffle=True)
            test_size = self.split_size["test"] / (self.split_size["valid"] + self.split_size["test"])
            valid_data, test_data = train_test_split(valid_test_data, test_size=test_size, random_state=0, shuffle=True)
            test_labels = torch.Tensor()

        # conditionalで指定するlabelをtrain用に取得する
        print('generate train data ...')
        for graph in tqdm(train_data):
            # クラスタ係数と最長距離を指定するためにパラメータを取得してlabelとする
            # paramsはリスト型で渡されるのでindex[0]をつける
            params = st.calc_graph_traits2csv([graph], self.condition_params)[0]
            tmp_label = []
            for param in params.values():
                tmp_label.append(round(param, self.condition_round))
            if len(tmp_label) == 1:
                tmp_label = torch.tensor(tmp_label).float().unsqueeze(0)
            else:
                tmp_label = torch.tensor(np.array([np.prod(tmp_label)])).float().unsqueeze(0)
            train_labels = torch.cat((train_labels, tmp_label), dim=0)
        train_labels.unsqueeze(1)

        # conditionalで指定するlabelをvalid用に取得する
        print('generate valid data ...')
        for graph in tqdm(valid_data):
            params = st.calc_graph_traits2csv([graph], self.condition_params)[0]
            tmp_label = []
            for param in params.values():
                tmp_label.append(round(param, self.condition_round))
            if len(tmp_label) == 1:
                tmp_label = torch.tensor(tmp_label).float().unsqueeze(0)
            else:
                tmp_label = torch.tensor(np.array([np.prod(tmp_label)])).float().unsqueeze(0)
            valid_labels = torch.cat((valid_labels, tmp_label), dim=0)
        valid_labels.unsqueeze(1)

        # test labelデータの作成
        if not self.split_size["test"] == 0:
            test_labels = torch.Tensor()
            print(f"generate test data ...")
            for graph in tqdm(test_data):
                params = st.calc_graph_traits2csv([graph], self.condition_params)[0]
                tmp_label = []
                for param in params.values():
                    tmp_label.append(round(param, self.condition_round))
                if len(tmp_label) == 1:
                    tmp_label = torch.tensor(tmp_label).float().unsqueeze(0)
                else:
                    tmp_label = torch.tensor(np.array([np.prod(tmp_label)])).float().unsqueeze(0)
                test_labels = torch.cat((test_labels, tmp_label), dim=0)
            test_labels.unsqueeze(1)

        # joblibでtrain, valid, testデータにラベルをconcatしたデータを保存する
        joblib.dump([train_data, train_labels], f"{self.base_dataset_path}{str(self.args['dataset'])}_train")
        joblib.dump([train_data, train_labels], f"./{self.result_dir}/dataset/{str(self.args['dataset'])}_train")
        joblib.dump([valid_data, valid_labels], f"{self.base_dataset_path}{str(self.args['dataset'])}_valid")
        joblib.dump([valid_data, valid_labels], f"./{self.result_dir}/dataset/{str(self.args['dataset'])}_valid")
        if not self.split_size["test"] == 0:
            joblib.dump([test_data, test_labels], f"{self.base_dataset_path}{str(self.args['dataset'])}_test")
            joblib.dump([test_data, test_labels], f"./{self.result_dir}/dataset/{str(self.args['dataset'])}_test")

    def create_disentanglement_er_dataset(self):
        """disentanglementを確かめるためのERランダムグラフと生成パラメータをconditionにしたデータセットを作成する関数
        
        - ERランダムグラフモデルの生成パラメータの1つであるエッジ確率pをconditionとする。
        """
        # グラフobjと生成パラメータを読み込み
        edge_files  = glob.glob(self.er_disentanglement_dataset_path + 'edge_*')
        node_files  = glob.glob(self.er_disentanglement_dataset_path + 'node_*')
        attr_files  = glob.glob(self.er_disentanglement_dataset_path + 'attr_*')
        edge_files, node_files, attr_files = sorted(edge_files), sorted(node_files), sorted(attr_files)

        # textからグラフobjへ変換する
        graph_data  = text2graph(edge_files, node_files=node_files)
        
        # 生成パラメータを取り出す
        attr_data   = []
        for attr_file in attr_files:
            with open(attr_file, "r") as f:
                attr_text   = f.read()
                n, p        = attr_text.split()
                attr_data.append((int(n), float(p)))

        # datasetをtrain, valid, testに分割する
        if self.split_size["test"] == 0:
            # when you do not need test data
            train_data, valid_data, train_attr, valid_attr = train_test_split(graph_data, attr_data,
                                                                                test_size=self.split_size["valid"],
                                                                                train_size=self.split_size["train"],
                                                                                random_state=0,
                                                                                shuffle=True)
        else:
            # when you need test data
            train_data, valid_test_data, train_attr, valid_test_attr = train_test_split(graph_data, attr_data,
                                                            test_size=self.split_size["valid"]+self.split_size["test"],
                                                            train_size=self.split_size["train"],
                                                            random_state=0, shuffle=True)
            test_size = self.split_size["test"] / (self.split_size["valid"] + self.split_size["test"])
            valid_data, test_data, valid_attr, test_attr = train_test_split(valid_test_data, valid_test_attr, test_size=test_size, random_state=0, shuffle=True)
            test_labels = torch.Tensor()

        # train dataとvalid dataのグラフ統計量の入れ物
        train_labels = torch.Tensor()
        valid_labels = torch.Tensor()

        # conditionalで指定するlabelをtrain用に取得する
        print('generate train data ...')
        for graph, attr in tqdm(zip(train_data, train_attr)):
            tmp_label = []
            tmp_label.append(round(attr[1], self.condition_round))
            tmp_label = torch.tensor(tmp_label).float().unsqueeze(0)
            train_labels = torch.cat((train_labels, tmp_label), dim=0)
        train_labels.unsqueeze(1)

        # conditionalで指定するlabelをvalid用に取得する
        print('generate valid data ...')
        for graph, attr in tqdm(zip(valid_data, valid_attr)):
            tmp_label = []
            tmp_label.append(round(attr[1], self.condition_round))
            tmp_label = torch.tensor(tmp_label).float().unsqueeze(0)
            valid_labels = torch.cat((valid_labels, tmp_label), dim=0)
        valid_labels.unsqueeze(1)

        # test labelデータの作成
        if not self.split_size["test"] == 0:
            test_labels = torch.Tensor()
            print(f"generate test data ...")
            for graph, attr in tqdm(zip(test_data, test_attr)):
                tmp_label = []
                tmp_label.append(round(attr[1], self.condition_round))
                tmp_label = torch.tensor(tmp_label).float().unsqueeze(0)
                test_labels = torch.cat((test_labels, tmp_label), dim=0)
            test_labels.unsqueeze(1)

        # joblibでtrain, valid, testデータにラベルをconcatしたデータを保存する
        joblib.dump([train_data, train_labels], f"./{self.result_dir}/dataset/{str(self.args['dataset'])}_train")
        joblib.dump([valid_data, valid_labels], f"./{self.result_dir}/dataset/{str(self.args['dataset'])}_valid")
        if not self.split_size["test"] == 0:
            joblib.dump([test_data, test_labels], f"./{self.result_dir}/dataset/{str(self.args['dataset'])}_test")

    def _create_dataset(self, detail, do_type='train'):
        """統計的手法や既存のデータセットから, グラフオブジェクトを作成する関数.

        Args:
            detail (dict): グラフオブジェクトの作成に関する詳細
            do_type (str, optional): 実行タイプ. 可視化するときは, "visualize"とする.

        Returns:
            (dict): keyは統計的手法や既存のデータセットの名称, valueはグラフオブジェクトのリスト.
        """
        datasets = {}
        for i, (key, value) in enumerate(detail.items()):
            generate_num = value[0]
            data_dim = value[1]
            params = value[2]

            params_list = []
            for param in params:
                if key == "twitter_pickup":
                    data = self.pickup_twitter_data(generate_num)
                else:
                    print("引数で指定されたdetailに無効なkeyが含まれているため, skipします.")

                # NNモデルでの生成時にはこっちを使う　いろんなparamのデータをまとめて一つのデータセットにするため
                if do_type == 'train':
                    params_list.extend(data)
                elif do_type == 'visualize':
                    # visualizeのみはこっちを使う　paramを分けてデータを分析したいため
                    params_list.append(data)
                else:
                    print("無効な do_type です.")
                    exit(1)
            datasets[key] = params_list
        return datasets

    def pickup_twitter_data(self, sampling_num):
        """Twitterデータセットからランダムに指定する数だけサンプリングする関数

        Args:
            sampling_num (int): Twitterデータセットからサンプリングする数

        Returns:
            (list): Twitterデータセットからサンプリングしたグラフのリスト
        """
        text_files = glob.glob(self.twitter_path)
        data = text2graph(text_files)
        sample_data = random.sample(data, sampling_num)
        return sample_data
    
    def pickup_reddit_data(self, sampling_num):
        """redditデータセットからランダムに指定する数だけサンプリングする関数
        
        Args:
            sampling_num (int): redditデータセットからサンプリングする数
            
            Returns:
                (list): redditデータセットからサンプリングしたグラフのリスト
        """
        sample_data = [] 
        text_files = open(self.reddit_path,'r')
        text_files = json.load(text_files)
        for i in range(len(text_files)):
            G = nx.Graph()
            G.add_edges_from(text_files[str(i)])
            sample_data.append(G)
        sample_data = random.sample(sample_data, k=sampling_num)
        return sample_data

    def pickup_origin_data(self, sampling_num):
        # [data,label]の形式で保存されているので、dataのみを呼び出す
        result_dir = str(Path(self.args['eval_graphs']).parent)
        graphs = joblib.load(f"{result_dir}/dataset/{str(self.args['dataset'])}_train")[0]
        sample_data = random.sample(graphs, sampling_num)
        return sample_data


if __name__ == "__main__":
       # 引数やGlobal変数を設定
    parser = argparse.ArgumentParser()
    parser = common_args(parser)
    args = parser.parse_args()
    params = Parameters(**utils.setup_params(vars(args), args.parameters))  # args，run_date，git_revisionなどを追加した辞書を取得
    complex_networks = ComplexNetworks(params=params)
    complex_networks.create_dataset()
    # complex_networks.make_apl_uniform_graph_with_label()