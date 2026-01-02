"""
よく使用されるグラフに関する処理をまとめたモジュール.

主な機能は以下.
・グラフオブジェクトから隣接行列への変換
・テキストファイルからグラフオブジェクトを作成
・グラフオブジェクトのグラフ特徴量を計算し, csvへ出力
"""

import networkx as nx
import numpy as np
from logging import getLogger
import random 

logger = getLogger(__name__)

def graph_obj2mat(G):
    """networkxのグラフオブジェクトを隣接行列に変換する関数
    
    同じラベルを持つノードはない前提.

    Args:
        G (networkx.classes.graph.Graph): networkxのグラフオブジェクト

    Returns:
        (numpy.ndarray) : 隣接行列
    """
    nodes = G.nodes
    edges = G.edges
    nodes = {i: node_label for i, node_label in enumerate(nodes)}

    adj_mat = np.zeros((len(nodes), len(nodes)))

    # forでぶん回している. smartにしたい
    for edge in edges:
        node1 = edge[0]
        node2 = edge[1]

        node1_arg = None
        node2_arg = None
        for key, node_label in nodes.items():
            if node1 == node_label:
                node1_arg = key
            if node2 == node_label:
                node2_arg = key

            # for短縮のため
            if not node1_arg is None and not node2_arg is None:
                break
        adj_mat[node1_arg, node2_arg] = 1
        adj_mat[node2_arg, node1_arg] = 1
    return adj_mat

def text2graph(text_files, node_files=None):
    """
    テキストファイルからグラフオブジェクトを作成する関数
    
        Args:
            text_files           (list) : text fileのリスト
            node_files (list, optional) : node idが記載されたtext fileのリスト. Defaults to None.
        
        Returns:
            (list) : グラフオブジェクトのリスト
    """
    graph_data = []
    
    if node_files is None:
        for text_file in text_files:
            with open(text_file, 'rb') as f:
                G = nx.read_edgelist(f, nodetype=int)
            graph_data.append(G)
    else:    
        for edge_file, node_file in zip(text_files, node_files):
            with open(edge_file, 'rb') as f:
                G = nx.read_edgelist(f, nodetype=int)
            with open(node_file, 'r') as f:
                node_text   = f.read()
                node_ids    = node_text.split()
                for node_id in node_ids:
                    G.add_node(int(node_id))
            graph_data.append(G)

    return graph_data

def dfs_code_to_graph_obj(dfs_code,end_value_list, edge_num=None, sampled_rank = None):
    """DFScodeをnetworkxのグラフオブジェクトに変換する関数

    Args:
        dfs_code ([np.array]): [(sequence,5)のnp.array]
        end_value_list ([list]): [終了コード[5]]
        sampled_rank    ([np.array]): [サンプリングで何番目の確率が選ばれたかを格納するnp.array]
    Returns:
        [networkx_graph]: [networkxのグラフオブジェクトを返す]
    """    
    G = nx.Graph()
    for t, current_code in enumerate(dfs_code):
        for i in range(len(current_code)):
            if edge_num is not None:
                if len(G.edges) >= edge_num:
                    return G
                elif current_code[i] == end_value_list[i]-1:
                    return G
            else:
                # 長さ自体はend_value_listの値だが実際の値は0から始まっているため-1する
                if current_code[i] == end_value_list[i]-1:
                    return G

        tuples = []
        tuples = current_code
        if sampled_rank == None:
            G.add_edge(tuples[0], tuples[1])
        else:
            color = "black" if ((sampled_rank[0][t] == 0) and (sampled_rank[1][t] == 0)) else "blue"
            G.add_edge(tuples[0], tuples[1], color=color)
    return G

def sort_nodes_by_degree(G):
    """次数が高い順にノード番号を付け替える関数

    Args:
        G (networkx.classes.graph.Graph): networkxのグラフオブジェクト

    Returns:
        (networkx.classes.graph.Graph) : 付け替え後のグラフオブジェクト
    """
    idx_deg_list = list(G.degree())
    sorted_idx_deg_list = sorted(idx_deg_list, key=lambda x: x[1])
    sorted_idx_deg_list.reverse()
    sorted_idx = [t[0] for t in sorted_idx_deg_list]
    sorted_G = nx.relabel_nodes(G, {v:u for u, v in enumerate(sorted_idx)})

    return sorted_G

def random_graph_edit(graph: nx.Graph, del_ratio: float) -> nx.Graph:
    ''' ある割合でエッジを削除し、連結になるようにエッジを追加する
        assert文でグラフが連結であることを保証する。
        Args:
            G (nx.Graph): 対象のグラフ
            del_ratio (float): graphのエッジ数に対する削除するエッジ数の割合。
        Returns:
            nx.Graph: エッジをランダムに削除、追加したグラフ
    '''
    G = graph.copy()
    assert 0 < del_ratio < 1, 'del_ratio must be between 0 and 1'
    N = int(G.number_of_edges() * del_ratio)
    logger.debug(f'Delete {N} edges and Add edges to make the graph connected')
    for _ in range(N):
        # Get a random edge and remove it
        edge = random.choice(list(G.edges()))
        G.remove_edge(*edge) # Remove the edge
        
    # 連結になるようにエッジを追加する
    while not nx.is_connected(G):
        # Get a random 2 nodes and if they are not connected, connect them
        a, b = random.sample(G.nodes(), 2)
        if not G.has_edge(a, b):
            G.add_edge(a, b)
            
    assert nx.is_connected(G), 'Graph is not connected'
    return G

if __name__ == "__main__":
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (1 ,3), (3, 4), (4, 5)])
    mat = graph_obj2mat(G)
    print(type(mat))
    print(mat)