"""
グラフ統計量(グラフ特徴量, グラフプロパティ)を算出するモジュール.

"""
from time import time_ns
from networkx.classes import graph
from networkx.readwrite import json_graph
import numpy as np
import networkx as nx
import torch
from collections import OrderedDict, Counter
import networkx.algorithms.approximation.treewidth as nx_tree
import networkx.algorithms.community as nx_comm
# import community
from community import community_louvain
import random
import matplotlib.pyplot as plt
import joblib
from scipy.optimize import curve_fit
import sympy as sym
from sympy.plotting import plot
import sys
import time
import math
import json
import glob
from tqdm import tqdm
import csv
import os
from sklearn.model_selection import train_test_split
import powerlaw
import pandas as pd
import argparse
from logging import getLogger
import pickle
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from graph_process.graph_utils import graph_obj2mat

from config import Parameters

logger = getLogger(__name__)
class GraphStatistic:
    """
    グラフデータからグラフ統計量を算出するクラス.
    """

    def __init__(self, params=None):
        # ログ設定
        if params is None:
            pass
        else:
            self.result_dir = str(Path(params.args['parameters']).parent)
            self.stdout_log_level = params.args['stdout_log_level']
            # logger_name = __name__+'.'+self.__class__.__name__
            logger_name = self.__class__.__name__
            logger.info("Start ; GraphStatistic.__init__()")
            
            self.dataset = params.args['dataset']
            self.power_degree_border_line = params.power_degree_border_line
            self.skip_calc_ged = params.args['skip_calc_ged']
            logger.debug("Finish : GraphStatistic.__init__()")
    
    # 以下、グラフ統計量を計算する関数群
    def degree_dist(self, graph):
        """隣接行列を入力として次数分布を作成し,べき指数を計算する関数

            Args:
                graph (networkx.classes.graph.Graph) : networkxのグラフオブジェクト

            Returns:
                (float) : 次数分布のべき指数
        
            Examples:
                >>> from config import Parameters
                >>> params = Parameters()
                >>> graph_statistic = GraphStatistic()
                >>> G = nx.Graph()
                >>> G.add_edges_from([(1, 2), (2, 3), (1, 3), (3, 4), (4, 5)])
                >>> print(graph_statistic.degree_dist(G))
                -0.7324867603589632
        """
        degree = list(dict(nx.degree(graph)).values())

        import collections
        power_degree = dict(collections.Counter(degree))
        power_degree = sorted(power_degree.items(), key=lambda x:x[0])
        x = []
        y = []
        
        for i in power_degree:
            num = i[0]
            amount = i[1]
            x.append(num)
            y.append(amount)
        y = np.array(y) / sum(y)#次数を確率化
        sum_prob = 0
        for index,prob in enumerate(y):
            sum_prob += prob
            if sum_prob >= self.power_degree_border_line:
                border_index = index + 1
                break

        x_log = np.log(np.array(x))
        y_log = np.log(np.array(y))

        x_split_plot = x_log[border_index:]
        y_split_plot = y_log[border_index:]
        param =  np.polyfit(x_split_plot,y_split_plot,1)
        return param[0]
    
    def power_law_alpha(self, graph):
        """
        入力グラフの次数分布のpower law coefficientを計算する

            Args:
                graph (networkx.classes.graph.Graph) : networkxのグラフオブジェクト

            Returns:
                (float) : power law coefficient
        """
        # 隣接行列を取得する
        A_in = graph_obj2mat(graph)
        # node id毎の次数を取得する
        degrees = A_in.sum(axis=0).flatten()
        # intに変換する
        degrees = degrees.astype(int)
        # 各次数の頻度を計算する
        degree_counts = Counter(degrees)
        # ノード数を取得する
        total_nodes = len(graph)
        # 次数の頻度を確率に変換する
        degree_distribution = {degree: count / total_nodes for degree, count in degree_counts.items()}
        # 次数で昇順にソートする
        degree_distribution = dict(sorted(degree_distribution.items()))
        # Fit
        fit = powerlaw.Fit(list(degree_distribution.values()), discrete=True, verbose=False)
        power = fit.power_law.alpha

        if np.isnan(power):
            return 0
        else:
            return power

    def cluster_coeff(self, graph):
        """平均クラスタ係数を計算する関数

        Args:
            graph (networkx.classes.graph.Graph) : networkxのグラフオブジェクト

        Returns:
            float: 平均クラスタ係数
        """
        #graph = np.array(graph)
        #graph = mat2graph_obj(graph)
        return nx.average_clustering(graph)

    def ave_dist(self, graph):
        """平均最短経路長を計算する関数

        Args:
            graph (networkx.classes.graph.Graph) : networkxのグラフオブジェクト

        Returns:
            (float) : 平均最短経路長
        """
        #graph = np.array(graph)
        #graph = mat2graph_obj(graph)
        return nx.average_shortest_path_length(graph)

    def ave_degree(self, graph):
        """平均次数を計算する関数

        Args:
            graph (networkx.classes.graph.Graph) : networkxのグラフオブジェクト

        Returns:
            (float) : 平均次数
        """
        degree_count = 0
        degree_dict = graph.degree()
        for node in degree_dict:
            node_num, node_degree = node
            degree_count += node_degree
        return degree_count/graph.number_of_nodes()

    def density(self, graph):
        """グラフの密度を求める

        Args:
            graph (nx.graph): [計算したいnetworkx型のグラフ]

        Returns:
            [float]: [密度の値]
        """
        try:
            return nx.density(graph)
        except:
            return 0

    # 未完成
    def clique(self, graph):
        """グラフ内の最大クリークの数を計算する関数

        Args:
            graph (networkx.classes.graph.Graph) : networkxのグラフオブジェクト

        Returns:
            (int) : グラフ内の最大クリークの数
        """
        return nx.graph_number_of_cliques(graph)

    def modularity(self, graph):
        """グラフのmodularityを求める関数

        Args:
            graph (nx.graph): 計算したいnetworkx型のグラフ

        Returns:
            float: modularityの値
        """
        partition = community_louvain.best_partition(graph)
        return community_louvain.modularity(partition,graph)

    def number_of_clusters(self, graph):
        """クラスタの数を計算する関数

        Args:
            graph (networkx.classes.graph.Graph) : networkxのグラフオブジェクト

        Returns:
            (int) : クラスタの数
        """
        clusters_dict = community_louvain.best_partition(graph)
        clusters_set = set(value for key, value in clusters_dict.items())
        return len(clusters_set)

    def largest_component_size(self, graph):
        """最大コンポーネントサイズを計算する関数

        Args:
            graph (networkx.classes.graph.Graph) : networkxのグラフオブジェクト

        Returns:
            (int) : 最大コンポーネントサイズ
        """
        largest_graph = max(nx.connected_components(graph), key=len)
        return len(largest_graph)

    @classmethod
    def largest_component_subgraph(cls, graph):
        """最大サイズのコンポーネントを計算する関数

        Args:
            graph (networkx.classes.graph.Graph) : networkxのグラフオブジェクト

        Returns:
            (networkx.classes.graph.Graph) : 最大サイズのコンポーネント
        """
        return max([graph.subgraph(c).copy() for c in nx.connected_components(graph)], key=len)

    def number_of_connected_component(self, graph):
        pass

    def degree_assortativity(self, graph):
        pass

    def reciprocity(self, graph):
        pass

    def maximum_of_shortest_path_lengths(self, graph):
        """最長の長さを持つ最短経路長の長さを求める関数

        Args:
            graph (nx.graph): 計算したいnetworkx型のグラフ

        Returns:
            int: 経路の長さ
        """
        max_shortest_path_length = 0
        path_dict = nx.shortest_path(graph)
        for node_num, paths in path_dict.items():
            for connect_node_num, path in paths.items():
                if len(path) >= max_shortest_path_length:
                    max_shortest_path_length = len(path)

        return max_shortest_path_length
        

    def degree_centrality(self, graph):
        '''
        グラフの各ノードの次数中心性を導出する
        
        Args:
            graph (nx.graph): 計算したいnetworkx型のグラフ
        
        Returns:
            (dict): グラフの各ノードの次数中心性のdict
        '''
        degree_centers = nx.degree_centrality(graph)
        return degree_centers

    def betweenness_centrality(self, graph):
        '''
        グラフの各ノードの媒介中心性を導出する
        
        Args:
            graph (nx.graph): 計算したいnetworkx型のグラフ
        
        Returns:
            (dict): グラフの各ノードの媒介中心性のdict
        '''
        betweenness_centers = nx.betweenness_centrality(graph)
        return betweenness_centers

    def closeness_centrality(self, graph):
        '''
        グラフの各ノードの近接中心性を導出する
        
        Args:
            graph (nx.graph): 計算したいnetworkx型のグラフ
        
        Returns:
            (dict): グラフの各ノードの近接中心性のdict
        '''
        closeness_centers = nx.closeness_centrality(graph)
        return closeness_centers
    
    def get_average_params(self, csv_path, save_dir):
        """生成したグラフの各パラメータごとの平均値を返すプログラム

        Args:
            csv_path (str): グラフごとのパラメータをもつcsvファイルのpath
            save_dir (str): 保存先のpath
        """
        file_name = os.path.splitext(os.path.basename(csv_path))[0]

        df = pd.read_csv(csv_path)
        average_df = df.mean()

        average_df.to_csv(save_dir + file_name + '.csv')
    
    @staticmethod
    def get_percentile_params(csv_path, save_dir, prob_list=(0.25, 0.5, 0.75)):
        """生成したグラフの各パラメータごとの中央値やパーセンタイルを返すプログラム

        Parameters
        ----------
        csv_path : str
            グラフごとのパラメータをもつcsvファイルのpath
        save_dir : str
            保存先のpath
        prob_list : list
            パーセンタイルを出力する確率の値
        """

        file_name = os.path.splitext(os.path.basename(csv_path))[0]

        df = pd.read_csv(csv_path)
        df_list = []
        for p in prob_list:
            df_list.append(df.quantile(p))
        percentile_df = pd.concat(df_list, axis=1)
        percentile_df.to_csv(f'{save_dir}{file_name}.csv')

    def get_graph(self, eval_graphs: str=None)->dict:
        '''Train / Valid / Generated Graphのリストを取得し、辞書に格納する関数
            Args:
                eval_graphs : str : 評価グラフのディレクトリパス
            Returns:
                graph_files : dict : Train / Valid / Generated Graphのリストを格納した辞書
        '''

        logger.debug('Get Graph File Path')

        graph_files = {}
        assert type(self.result_dir) == str, f'Type of result_dir should be str, but {type(self.result_dir)}'
        dataset_files = sorted(glob.glob(self.result_dir+f'/dataset/{self.dataset}*'))
        assert len(dataset_files) == 2, f'Number of Dataset Files should be 2, but {len(dataset_files)}' 

        for dataset_file in dataset_files:
            logger.debug(f'Get Dataset Files from {dataset_file}')
            with open(dataset_file, 'rb') as f:
                dataset = pickle.load(f)
                assert len(dataset) == 2, f'Number of Dataset should be 2, but {len(dataset)}'
                assert type(dataset[0]) == list, f'Type of Dataset should be list, but {type(dataset[0])}'
                assert type(dataset[0][0]) == nx.Graph, f'Type of Dataset should be nx.Graph, but {type(dataset[0][0])}'

                graph_files[dataset_file.split('/')[-1]] = dataset[0]

        if eval_graphs is None:
            generated_files = sorted(glob.glob(self.result_dir+'/eval/*.pkl'))
        else:
            generated_files = sorted(glob.glob(eval_graphs + '*.pkl'))
        
        for generated_file in generated_files:
            logger.debug(f'Get Generate Files from {generated_file}')
            with open(generated_file, 'rb') as f:
                generated_graphs = pickle.load(f)
                assert type(generated_graphs) == list, f'Type of Generated Graphs should be list, but {type(generated_graphs)}'
                assert type(generated_graphs[0]) == nx.Graph, f'Type of Generated Graphs should be nx.Graph, but {type(generated_graphs[0])}'

                graph_files['.'.join(generated_file.split('/')[-1].split('.')[:-1])] = generated_graphs

        assert type(graph_files) == dict, f'Type of Graph Files should be dict, but {type(graph_files)}'
        graph_files = OrderedDict(sorted(graph_files.items(), key=lambda t: t[0])) # Sort by key
        return graph_files

    def get_quality(self, save_dir: str, eval_graphs: str=None)->None:
        '''Visualize時にQualityを出力するための関数
            Qualityは以下の二つの要素から判定される
            - Novelty : 生成グラフのうち、訓練グラフと異なるものの個数
            - Uniqueness : 生成グラフ同士で互いに異なるもの個数
            Args:
                save_dir: str : 結果を保存するディレクトリ
                eval_graphs: str : 評価グラフのディレクトリパス
        '''
        self.get_uniqueness(save_dir, eval_graphs=eval_graphs)
        self.get_novelty(save_dir, eval_graphs=eval_graphs)
        
    def get_uniqueness(self, save_dir: str, eval_graphs: str=None)->None:
        '''Visualize時にUniquenessを出力するための関数
            Args:
                save_dir: str : 結果を保存するディレクトリ
                eval_graphs: str : 評価グラフのディレクトリパス
        '''
        graph_files = self.get_graph(eval_graphs) # 入力グラフと生成グラフのリストの辞書
        uniqueness_df = pd.DataFrame() # Uniquenessを格納するdataframe
        columns_list = ['Input Size', 'Uniqueness', 'Ratio[%]'] # dataframeのカラム名
        for name,graph_list in graph_files.items():
            N = len(graph_list) # 生成したグラフの数
            # 生成したグラフのUniquenessを計算
            M = self.uniquness_test(graph_list) # Uniquenessの計算
            data = np.array([N, M, M/N*100], dtype=np.int) # dataframeに格納するデータ
            uniqueness_df = uniqueness_df.append(pd.DataFrame(data.reshape(1, -1), columns=columns_list, index=[name]))
        # dataframeをcsvファイルとして保存
        uniqueness_df.to_csv(os.path.join(save_dir, 'uniqueness.csv'))
    
    def get_novelty(self, save_dir: str, eval_graphs: str=None)->None:
        '''Visualize時にNoveltyを出力するための関数
            Args:
                save_dir: str : 結果を保存するディレクトリ
                eval_graphs: str : 評価グラフのディレクトリパス
        '''
        graph_files = self.get_graph(eval_graphs) # 入力グラフと生成グラフのリストの辞書
        novelty_df = pd.DataFrame() # Noveltyを格納するdataframe
        columns_list = ['Train Size', 'Generated/Valid Size', 'Novelty', 'Ratio[%]'] # dataframeのカラム名
        train_hashes = [] # 訓練グラフのハッシュ値を格納するリスト
        for graph in graph_files[self.dataset+'_train']:
            hash = nx.weisfeiler_lehman_graph_hash(graph)
            train_hashes.append(hash)
        N1 = len(train_hashes) # 訓練グラフの数
        for name,graph_list in graph_files.items():
            N2 = len(graph_list) # 検証データまたは生成したグラフの数
            if name==self.dataset+'_train':
                continue
            M = self.novelty_test(train_hashes, graph_list) # Noveltyの計算
            data = np.array([N1, N2, M, M/N2*100], dtype=np.int) # dataframeに格納するデータ
            novelty_df = novelty_df.append(pd.DataFrame(data.reshape(1, -1), columns=columns_list, index=[name]))
        # dataframeをcsvファイルとして保存
        novelty_df.to_csv(os.path.join(save_dir, 'novelty.csv'))

    @staticmethod
    def novelty_test(train_hashes: list, graph_list: list)->int:
        '''与えられたハッシュ値の集合とグラフのリストから、Noveltyを計算する関数
            Args:
                train_hashes: list : 訓練グラフのハッシュ値のリスト
                graph_list: list : 生成したグラフのリスト
            Returns:
                M: int : Noveltyの値
            '''
        M = 0 # Novelty
        for graph in graph_list:
            hash = nx.weisfeiler_lehman_graph_hash(graph)
            if hash in train_hashes:
                continue
            else:
                M+=1
        return M
    
    @staticmethod
    def uniquness_test(graph_list: list)->int:
        '''与えられたグラフのリストから、Uniquenessを計算する関数
            Uniquenessとは、生成したグラフ同士がどれだけ異なるかを表す指標である。
            最小値は、生成したグラフ同士が全く同じである場合に1となる。
            また、最大値は、生成したグラフ同士が全く異なる場合にlen(graph_list)となる。
            Args:
                graph_list: list : 生成したグラフのリスト
            Returns:
                uniqueness: float : Uniqueなグラフの個数
        '''
        hash_set = set() # 生成したグラフのハッシュ値を格納するset
        for graph in graph_list:
            hash = nx.weisfeiler_lehman_graph_hash(graph)
            hash_set.add(hash)
        return len(hash_set)
    
    def get_l2_norm_params(self, condition_params:list, condition_values:dict, csv_path:str, save_dir:str) -> None:
        """生成したグラフの各パラメータごとの平均を取ったL2 Norm(RMSE)を返すプログラム

        計算式:
            L2 Norm = {(E[x] - condition_value)^2 + Std[x]^2}^0.5
                    = {(E[x] - condition_value)^2 + Var[x]}^0.5
                    = {E[x - condition_value]^2 + Var[x]}^0.5
                    = {E[(x - condition_value)^2]}^0.5

        Args:
            condition_params (list): conditionとして与えている特徴量の名前のlist
            condition_values  (dict): 各特徴量がconditionとして持つ値のlistを格納したdict
            csv_path (str): グラフごとのパラメータをもつcsvファイルのpath
            save_dir (str): 保存先のpath
        """
        file_name = os.path.splitext(os.path.basename(csv_path))[0]

        df = pd.read_csv(csv_path)
        l2_norm_df = pd.DataFrame()

        for condition_name in condition_params:
            for condition_val in condition_values[condition_name]:
                l2_norm_df[f"{condition_name}_{condition_val}"] = pd.Series(np.sqrt(np.square(df[condition_name] - condition_val).mean()))

        l2_norm_df.to_csv(save_dir + file_name + '.csv')

    def calc_graph_traits2csv(self, graphs, eval_params):
        '''
        グラフごとにeval_paramsで指定されている特性値を計算してcsvへの保存形式に変換する関数

        Args:
            graphs: [graph_obj, ....]
            eval_params: 計算を行う特性値の名前のlist

        Returns:
            trait_list(list): 各グラフのparamのdictのlist
        '''
        trait_list=[]
        for index, graph in enumerate(graphs):
            tmp_dict = {}
            for key in eval_params:
                #if "id" in key:
                #    param = index
                if "Power-law exponent" in key:
                    try:
                        # param = self.degree_dist(graph)
                        param = self.power_law_alpha(graph)
                    except:
                        param = None
                if "Clustering coefficient" in key:
                    try:
                        param = self.cluster_coeff(graph)
                    except:
                        param = None
                if "Average path length" in key:
                    try:
                        param = self.ave_dist(graph)
                    except Exception as e:
                        print(e)
                        param = None
                if "Average degree" in key:
                    try:
                        param = self.ave_degree(graph)
                    except:
                        param = None
                if "Edge density" in key:
                    try:
                        param = self.density(graph)
                    except:
                        param = None
                if "Modularity" in key:
                    try:
                        param = self.modularity(graph)
                    except:
                        param = None
                if "Diameter" in key:
                    try:
                        param = self.maximum_of_shortest_path_lengths(graph)
                    except:
                        param = None
                if "degree_centrality" in key:
                    try:
                        param = self.degree_centrality(graph)
                    except:
                        param = None
                if "betweenness_centrality" in key:
                    try:
                        param = self.betweenness_centrality(graph)
                    except:
                        param = None
                if "closeness_centrality" in key:
                    try:
                        param = self.closeness_centrality(graph)
                    except:
                        param = None
                if "Largest component size" in key:
                    try:
                        param = self.largest_component_size(graph)
                    except:
                        param = None
                if "size" in key:
                    try:
                        param = graph.number_of_nodes()
                    except:
                        param = None
                tmp_dict.update({key:param})
            trait_list.append(tmp_dict)
        return trait_list

    def graph2csv(self, graphs, csv_dir, file_name, eval_params):
        '''各グラフデータのパラメータをcsvファイルに出力する関数

        Args:
            graphs      (list): グラフデータが格納されているリスト [GraphObj, ...]
            csv_dir   (string): csvファイルをの保存先ディレクトリ
                                e.g. csv_dir = "result/2021_0101/visualize/csv/"
            file_name (string): csvファイル名
                                e.g. file_name = "AveragePathLength_01"
            eval_params (list): 計算したいグラフ特徴量の名称のリスト
                                e.g. eval_params = ["Power-law exponent", "Clustering coefficient"]
        '''
        trait_dict = self.calc_graph_traits2csv(graphs, eval_params)
        with open(csv_dir + file_name + '.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=eval_params)
            writer.writeheader()
            writer.writerows(trait_dict)

    def unconnected_ratio(self, unconnected_ratio_dict, unconnected_ratio_dir):
        '''グラフの非連結率を計算する関数

        Args:
            unconnected_ratio_dict (dict): ファイル名 : (グラフの非連結率, 非連結グラフの総数)の形式で格納した辞書
            unconnected_ratio_dir (str): 非連結率を保存するディレクトリ
        '''
        with open(unconnected_ratio_dir + 'unconnected_ratio.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['file_name', 'unconnected_ratio', 'total'])
            for key, value in sorted(unconnected_ratio_dict.items(), key=lambda x:x[0]):
                writer.writerow([key, value[0], value[1]])

if __name__ == "__main__":
    from config import Parameters
    params = Parameters()

    graph_statistic = GraphStatistic()
    # G = nx.Graph()
    # print(type(G))
    # G.add_edges_from([(1, 2), (2, 3), (1, 3), (3, 4), (4, 5)])
    # trait_list = graph_statistic.calc_graph_traits2csv([G], params.eval_params)
    # print(trait_list)

    G = nx.barabasi_albert_graph(4, 1)

    graph_statistic.power_law_alpha(G)