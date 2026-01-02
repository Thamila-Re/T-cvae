"""
networkx形式のグラフオブジェクトからシーケンスコードへ変換するモジュール.
"""

from lib2to3.pytree import convert
from unicodedata import name
import networkx as nx
from collections import OrderedDict
import numpy as np
from collections import deque
import random


class ConvertToSequenceCode():
    """
    グラフをDFSコードへ変換するクラス.
    """
    
    def __init__(self, graph, mode="normal", search_type="dfs", use_transition_flag = False):
        """
        Args:
            graph (nx.graph) : networkx形式のグラフオブジェクト
            mode (str) : ノードを選択する際の挙動
                You can select from ["normal", "low_degree_first", "high_degree_first"].
                default: "normal"
            search_type (str) : 探索方法
                You can select from ["dfs", "bfs", "rfs"].
                default: "dfs"
            use_transition_flag (bool) : 遷移フラグを使うかどうか
                default: False
        """
        self.G = graph
        self.node_tree = [node for node in graph.nodes()]
        self.edge_tree = [edge for edge in graph.edges()]
        self.seq_code = list()
        self.visited_edges = list()
        self.time_stamp = 0
        self.node_time_stamp = [-1 for i in range(graph.number_of_nodes())]
        self.mode=mode
        self.visit_queue = deque()
        if search_type == "bfs" or search_type == "dfs":
            self.visit_queue.append(self.get_max_degree_index())
        else:
            self.visit_queue.append(random.choice(list(self.G.nodes)))

        self.search_type = search_type
        self.use_transition_flag = use_transition_flag

    def get_max_degree_index(self):
        max_degree = 0
        max_degree_index = 0
        for i in range(self.G.number_of_nodes()):
            if(self.G.degree(i) >= max_degree):
                max_degree = self.G.degree(i)
                max_degree_index = i

        return max_degree_index

    def dfs(self,current_node):
        neightbor_node_dict = OrderedDict({neightbor:self.node_time_stamp[neightbor] for neightbor in self.G.neighbors(current_node)})
        neighbor_degree_dict = OrderedDict({neighbor: self.G.degree[neighbor] for neighbor in neightbor_node_dict.keys()})
        if self.mode=="high_degree_first":
            # degreeの値でsort
            sorted_neighbor_degree = OrderedDict(sorted(neighbor_degree_dict.items(), key=lambda x: x[1], reverse=True))
            # neighborのnode idをdegreeで並び替え
            sorted_neightbor_node = {key: neightbor_node_dict[key] for key in sorted_neighbor_degree.keys()}
        elif self.mode=="low_degree_first":
            # degreeの値でsort
            sorted_neighbor_degree = OrderedDict(sorted(neighbor_degree_dict.items(), key=lambda x: x[1], reverse=False))
            # neighborのnode idをdegreeで並び替え
            sorted_neightbor_node = {key: neightbor_node_dict[key] for key in sorted_neighbor_degree.keys()}
        else:
            sorted_neightbor_node = OrderedDict(sorted(neightbor_node_dict.items(), key=lambda x: x[1], reverse=True))

        if(len(self.visited_edges) == len(self.edge_tree)):
            return

        edge_id = 0
        for next_node in sorted_neightbor_node.keys():
            # visited_edgesにすでに訪れたエッジの組み合わせがあったらスルー
            if((current_node, next_node) in self.visited_edges or (next_node, current_node)in self.visited_edges):
                continue
            else:
                if(self.node_time_stamp[next_node] != -1):
                    # 現在のノードにタイムスタンプが登録されていなければタイムスタンプを登録
                    if(self.node_time_stamp[current_node] == -1):
                        self.node_time_stamp[current_node] = self.time_stamp
                        self.time_stamp += 1
                        edge_id = 1 if self.use_transition_flag else 0

                    self.visited_edges.append((current_node,next_node))
                    # print(f"{current_node} => {next_node}")
                    self.seq_code.append([self.node_time_stamp[current_node],self.node_time_stamp[next_node],self.G.degree(current_node),self.G.degree(next_node),edge_id])
                else:
                    # 現在のノードにタイムスタンプが登録されていなければタイムスタンプを登録
                    if(self.node_time_stamp[current_node] == -1):
                        self.node_time_stamp[current_node] = self.time_stamp
                        self.time_stamp += 1
                        edge_id = 1 if self.use_transition_flag else 0
                    # 次のノードにタイムスタンプが登録されていなければタイムスタンプを登録
                    if(self.node_time_stamp[next_node] == -1):
                        self.node_time_stamp[next_node] = self.time_stamp
                        self.time_stamp += 1
                    # timeStamp_u, timeStamp_v, nodeLabel u, nodeLable_v ,edgeLable(u,v)の順のリストを作成
                    # print(f"{current_node} => {next_node}")
                    self.seq_code.append([self.node_time_stamp[current_node],self.node_time_stamp[next_node],self.G.degree(current_node),self.G.degree(next_node),edge_id])
                    self.visited_edges.append((current_node,next_node))
                    self.dfs(next_node)
                    edge_id = 1 if self.use_transition_flag else 0

    def bfs(self):
        current_node = self.visit_queue.popleft()
        neightbor_node_dict = OrderedDict({neightbor:self.node_time_stamp[neightbor] for neightbor in self.G.neighbors(current_node)})
        # if len(neightbor_node_dict) == 1:
        #     return
        neighbor_degree_dict = OrderedDict({neighbor: self.G.degree[neighbor] for neighbor in neightbor_node_dict.keys()})
        if self.mode=="high_degree_first":
            # degreeの値でsort
            sorted_neighbor_degree = OrderedDict(sorted(neighbor_degree_dict.items(), key=lambda x: x[1], reverse=True))
            # neighborのnode idをdegreeで並び替え
            sorted_neightbor_node = {key: neightbor_node_dict[key] for key in sorted_neighbor_degree.keys()}
        elif self.mode=="low_degree_first":
            # degreeの値でsort
            sorted_neighbor_degree = OrderedDict(sorted(neighbor_degree_dict.items(), key=lambda x: x[1], reverse=False))
            # neighborのnode idをdegreeで並び替え
            sorted_neightbor_node = {key: neightbor_node_dict[key] for key in sorted_neighbor_degree.keys()}
        else:
            sorted_neightbor_node = OrderedDict(sorted(neightbor_node_dict.items(), key=lambda x: x[1], reverse=True))

        if(len(self.visited_edges) == len(self.edge_tree)):
            return

        edge_id = 1 if self.use_transition_flag else 0
        for next_node in sorted_neightbor_node.keys():
            # visited_edgesにすでに訪れたエッジの組み合わせがあったらスルー
            if((current_node, next_node) in self.visited_edges or (next_node, current_node)in self.visited_edges):
                continue
            else:
                self.visit_queue.append(next_node)
                if(self.node_time_stamp[next_node] != -1):
                    # 現在のノードにタイムスタンプが登録されていなければタイムスタンプを登録
                    if(self.node_time_stamp[current_node] == -1):
                        self.node_time_stamp[current_node] = self.time_stamp
                        self.time_stamp += 1

                    self.visited_edges.append((current_node,next_node))
                    # print(f"{current_node} => {next_node}")
                    # self.seq_code.append([self.node_time_stamp[current_node],self.node_time_stamp[next_node], current_node , next_node,0])
                    self.seq_code.append([self.node_time_stamp[current_node],self.node_time_stamp[next_node],self.G.degree(current_node),self.G.degree(next_node),edge_id])
                    edge_id = 0

                else:
                    # 現在のノードにタイムスタンプが登録されていなければタイムスタンプを登録
                    if(self.node_time_stamp[current_node] == -1):
                        self.node_time_stamp[current_node] = self.time_stamp
                        self.time_stamp += 1
                    # 次のノードにタイムスタンプが登録されていなければタイムスタンプを登録
                    if(self.node_time_stamp[next_node] == -1):
                        self.node_time_stamp[next_node] = self.time_stamp
                        self.time_stamp += 1
                    # timeStamp_u, timeStamp_v, nodeLabel u, nodeLable_v ,edgeLable(u,v)の順のリストを作成
                    # print(f"{current_node} => {next_node}")
                    # self.seq_code.append([self.node_time_stamp[current_node],self.node_time_stamp[next_node], current_node , next_node,0])
                    self.seq_code.append([self.node_time_stamp[current_node],self.node_time_stamp[next_node],self.G.degree(current_node),self.G.degree(next_node),edge_id])
                    self.visited_edges.append((current_node,next_node))
                    edge_id = 0

        self.bfs()

    def rfs(self):
        current_node = random.choice(list(self.visit_queue))
        self.visit_queue.remove(current_node)
        neightbor_node_dict = OrderedDict({neightbor:self.node_time_stamp[neightbor] for neightbor in self.G.neighbors(current_node)})
        # if len(neightbor_node_dict) == 1:
        #     return
        neighbor_degree_dict = OrderedDict({neighbor: self.G.degree[neighbor] for neighbor in neightbor_node_dict.keys()})
        if self.mode=="high_degree_first":
            # degreeの値でsort
            sorted_neighbor_degree = OrderedDict(sorted(neighbor_degree_dict.items(), key=lambda x: x[1], reverse=True))
            # neighborのnode idをdegreeで並び替え
            sorted_neightbor_node = {key: neightbor_node_dict[key] for key in sorted_neighbor_degree.keys()}
        elif self.mode=="low_degree_first":
            # degreeの値でsort
            sorted_neighbor_degree = OrderedDict(sorted(neighbor_degree_dict.items(), key=lambda x: x[1], reverse=False))
            # neighborのnode idをdegreeで並び替え
            sorted_neightbor_node = {key: neightbor_node_dict[key] for key in sorted_neighbor_degree.keys()}
        else:
            sorted_neightbor_node = OrderedDict(sorted(neightbor_node_dict.items(), key=lambda x: x[1], reverse=True))

        if(len(self.visited_edges) == len(self.edge_tree)):
            return

        edge_id = 1 if self.use_transition_flag else 0
        for next_node in sorted_neightbor_node.keys():
            # visited_edgesにすでに訪れたエッジの組み合わせがあったらスルー
            if((current_node, next_node) in self.visited_edges or (next_node, current_node)in self.visited_edges):
                continue
            else:
                self.visit_queue.append(next_node)
                if(self.node_time_stamp[next_node] != -1):
                    # 現在のノードにタイムスタンプが登録されていなければタイムスタンプを登録
                    if(self.node_time_stamp[current_node] == -1):
                        self.node_time_stamp[current_node] = self.time_stamp
                        self.time_stamp += 1

                    self.visited_edges.append((current_node,next_node))
                    # print(f"{current_node} => {next_node}")
                    # self.seq_code.append([self.node_time_stamp[current_node],self.node_time_stamp[next_node], current_node , next_node,0])
                    self.seq_code.append([self.node_time_stamp[current_node],self.node_time_stamp[next_node],self.G.degree(current_node),self.G.degree(next_node),edge_id])
                    edge_id = 0

                else:
                    # 現在のノードにタイムスタンプが登録されていなければタイムスタンプを登録
                    if(self.node_time_stamp[current_node] == -1):
                        self.node_time_stamp[current_node] = self.time_stamp
                        self.time_stamp += 1
                    # 次のノードにタイムスタンプが登録されていなければタイムスタンプを登録
                    if(self.node_time_stamp[next_node] == -1):
                        self.node_time_stamp[next_node] = self.time_stamp
                        self.time_stamp += 1
                    # timeStamp_u, timeStamp_v, nodeLabel u, nodeLable_v ,edgeLable(u,v)の順のリストを作成
                    # print(f"{current_node} => {next_node}")
                    # self.seq_code.append([self.node_time_stamp[current_node],self.node_time_stamp[next_node], current_node , next_node,0])
                    self.seq_code.append([self.node_time_stamp[current_node],self.node_time_stamp[next_node],self.G.degree(current_node),self.G.degree(next_node),edge_id])
                    self.visited_edges.append((current_node,next_node))
                    edge_id = 0
        self.rfs()

    def get_seq_code(self):
        if self.search_type == 'dfs':
            self.dfs(self.get_max_degree_index())
        elif self.search_type == 'bfs':
            self.bfs()
        elif self.search_type == 'rfs':
            self.rfs()
        return np.array(self.seq_code)


if __name__ == "__main__":
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (0 ,2), (2, 3), (3, 4), (1, 5), (5, 6)])
    converter = ConvertToSequenceCode(graph=G, mode="high_degree_first", search_type='rfs', use_transition_flag=False)
    seq_code = converter.get_seq_code()
    print(seq_code)
