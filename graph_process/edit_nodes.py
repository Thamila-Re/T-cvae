"""
グラフオブジェクトのエッジを追加あるいは削除するためのモジュール.
"""
import networkx as nx


def remove_degree_is_one_nodes(graphs):
    """
    グラフオブジェクトから次数が1のノードを削除する関数.
    Args:
        graphs (list) : 編集前のグラフオブジェクトのリスト

    Returns:
        (list) : 編集後のグラフオブジェクトのリスト

    Examples:
        >>> G = nx.Graph()
        >>> G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> G.add_edges_from([(1, 2),(2, 4),(3, 2),(4, 5),(6, 10),(7, 6),(8, 6),(9, 8),(10, 8),(1, 9),(8, 1),])
        >>> graph = remove_degree_is_one_nodes([G])
        >>> print(graph[0].degree)
        [(1, 3), (2, 2), (4, 1), (6, 2), (8, 4), (9, 2), (10, 2)]
    """
    edited_graphs = list()

    for graph in graphs:
        while True:
            nodes_to_remove = [n for n in graph if graph.degree(n) <=1]
            if len(nodes_to_remove) == 0:
                break
            graph.remove_nodes_from(nodes_to_remove)
        edited_graphs.append(graph)

    return edited_graphs

def relabel_graphs_nodes(graphs):
    """
    グラフオブジェクトのラベルを0〜ノード数に振り直す関数.
    Args:
        graphs (list) : 編集前のグラフオブジェクトのリスト

    Returns:
        (list) : 編集後のグラフオブジェクトのリスト
    Examples:
        >>> G = nx.Graph()
        >>> G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> G.add_edges_from([(1, 2),(2, 4),(3, 2),(4, 5),(6, 10),(7, 6),(8, 6),(9, 8),(10, 8),(1, 9),(8, 1),])
        >>> graphs = remove_degree_is_one_nodes([G])
        >>> graphs = relabel_graphs_nodes(graphs)
        >>> print(graphs[0].degree)
        [(0, 3), (1, 2), (2, 1), (3, 2), (4, 4), (5, 2), (6, 2)]
    """
    mapping = dict()
    edited_graphs = list()

    for graph in graphs:
        mapping = {k: v for k, v in zip(graph.nodes, list(range(nx.number_of_nodes(graph))))}
        graph = nx.relabel_nodes(graph, mapping)
        edited_graphs.append(graph)
    
    return edited_graphs

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    