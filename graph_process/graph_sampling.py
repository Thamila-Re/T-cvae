"""
networkx形式のグラフオブジェクトからDFS code,BFS code以外のシーケンスコードへ変換するモジュール.
"""

import networkx as nx
from collections import OrderedDict
import numpy as np
import queue

class GraphSampling():
    """
    グラフをDFS code,BFS code以外のシーケンスコードへ変換するクラス.
    """

    def __init__(self, graph, node_order="normal", search_type="2nd_rw", p=1.0, q=1.0):
        """
        Args:
            graph (nx.graph) : networkx形式のグラフオブジェクト.
            search_type (str) : 探索方法
                default: "2nd_rw"
            p (float) : 2nd order random walkのバイアスパラメータp.
                default: 1.0
            q (float) : 2nd order random walkのバイアスパラメータq.
                default: 1.0
        """
        self.G=graph
        self.node_tree = [node for node in graph.nodes()]
        self.edge_tree = [edge for edge in graph.edges()]

        self.node_order=node_order
        self.search_type = search_type

        self.p=p
        self.q=q

    def second_order_randomwalk(self, first_node, edge_num=-1):
        """2nd order random walkを行う関数

        Args:
            first_node (int): 最初のノードのid.
            edge_num (int, optional): サンプリング時のエッジ数上限. 
                default: -1
        """
        def calculate_alpha_pq(previous_node, neighbor_node):
            d_tx = nx.shortest_path_length(self.G, previous_node, neighbor_node)
            if(d_tx==0):
                return 1.0/self.p
            elif(d_tx==1):
                return 1.0
            else:
                return 1.0/self.q

        seq_code = list()
        visited_edges = list()
        time_stamp = 0
        node_time_stamp = [-1 for i in range(self.G.number_of_nodes())]
        node_time_stamp[first_node] = time_stamp
        time_stamp += 1
        neightbors = list()
        alpha_pq = dict()
        previous_node = -1
        current_node = first_node
        # nodes_prob = list()
        preprocessed_weights = dict()
        edge_list = self.G.edges()
        for edge in edge_list:
            for e_num, p_node in enumerate(edge):
                neightbors = self.G.neighbors(edge[not e_num])
                for n in neightbors:
                    preprocessed_weights[(p_node, edge[not e_num], n)] = calculate_alpha_pq(p_node, n)
                    preprocessed_weights[(-1, edge[not e_num], n)] = 1.0
        while len(visited_edges) != len(self.edge_tree) and len(visited_edges) != edge_num:
            alpha_pq = {key[2]:value for key, value in preprocessed_weights.items() if key[0]==previous_node and key[1]==current_node}
            previous_node = current_node
            current_node = list(alpha_pq.keys())[sample_alias_method(list(alpha_pq.values()))[0]]
            if node_time_stamp[current_node] == -1:
                node_time_stamp[current_node] = time_stamp
                time_stamp += 1
            if (previous_node, current_node) in visited_edges or (current_node, previous_node) in visited_edges:
                continue
            else:
                visited_edges.append((previous_node, current_node))
                seq_code.append([node_time_stamp[previous_node], node_time_stamp[current_node], self.G.degree(previous_node), self.G.degree(current_node), 0])
            alpha_pq.clear()

        return seq_code

    def get_max_degree_index(self):
        max_degree = 0
        max_degree_index = 0
        for i in range(self.G.number_of_nodes()):
            if(self.G.degree(i) >= max_degree):
                max_degree = self.G.degree(i)
                max_degree_index = i

        return max_degree_index

    def get_min_degree_index(self):
        min_degree = 0
        min_degree_index = 0
        for i in range(self.G.number_of_nodes()):
            if(self.G.degree(i) <= min_degree):
                min_degree = self.G.degree(i)
                min_degree_index = i

        return min_degree_index
    
    def get_random_node_index(self):
        return np.random.choice(list(self.G.nodes))
    
    def adj_matrix(self):
        ordered_degree = self.G.degree
        if self.node_order == "high_degree_first" or self.node_order == "low_degree_first":
            ordered_degree = sorted(self.G.degree, key=lambda x: x[1], reverse=True if self.node_order == "high_degree_first" else False)
        else:
            pass
        return nx.adjacency_matrix(self.G, nodelist=[n[0] for n in ordered_degree]).todense().astype(int) # type: ignore
    
    def get_seq_code(self):
        """Function to get sequence code by graph sampling.

        Returns:
            ndarray[Unknown, Unknown]: Numpy array of sequence code.
        """
        if(self.search_type == "2nd_rw"):
            first_node = 0
            if(self.node_order == "normal"):
                first_node = self.get_random_node_index()
            elif(self.node_order == "high_degree_first"):
                first_node = self.get_max_degree_index()
            elif(self.node_order == "low_degree_first"):
                first_node = self.get_min_degree_index()
            return np.array(self.second_order_randomwalk(first_node))
        else:
            return np.array(self.second_order_randomwalk(0)) # Pylanceの注意表示が鬱陶しいので仮の表示
        
    def sample_from_dataset(self, edge_num, pre_calculate = False):
        if(self.search_type == "2nd_rw"):
            first_node = 0
            if(self.node_order == "normal"):
                first_node = self.get_random_node_index()
            elif(self.node_order == "high_degree_first"):
                first_node = self.get_max_degree_index()
            elif(self.node_order == "low_degree_first"):
                first_node = self.get_min_degree_index()
            return np.array(self.second_order_randomwalk(first_node, edge_num=edge_num))
        else:
            return np.array(self.second_order_randomwalk(0)) # Pylanceの注意表示が鬱陶しいので仮の表示

def sample_alias_method(wgt, nn=1):
    """Walker's Alias Method"""
    if not isinstance(wgt, np.ndarray):
        wgt = np.array(wgt)
    wsm = sum(wgt)
    n = len(wgt)
    p = (wgt*n/wsm).tolist()
    a, hl = [0] * n, [0] * n
    l, h = 0, n-1
    for i in range(n):
        if p[i] < 1:
            hl[l] = i
            l += 1
        else:
            hl[h] = i
            h -= 1
    while l > 0 and h < n-1:
        j, k = hl[l-1], hl[h+1]
        a[j] = k
        p[k] += p[j] - 1
        if p[k] < 1:
            hl[l-1] = k
            h += 1
        else:
            l -= 1
    rr = np.random.rand(nn) * n
    ii = np.int32(np.floor(rr))
    rr -= ii
    return np.array([i if r < p[i] else a[i] for i, r in zip(ii, rr)]) # type: ignore


if __name__ == "__main__":
    from graph_utils import text2graph
    text_files = ["./data/edgelists_50/renum_0.txt"]
    graph_data = text2graph(text_files)
    G = graph_data[0]
    converter = GraphSampling(graph=G, node_order="high_degree_first", p=4.0, q=0.25)
    print(converter.get_seq_code(True))