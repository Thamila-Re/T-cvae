"""
可視化のバックエンド側の処理全般がまとめられたモジュール.
"""

import os
from turtle import width
import matplotlib
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation
import numpy as np
import glob
import joblib
import random
import cv2
import shutil
# import time
# import ffmpeg

import utils


def scatter_diagram_visualize(eval_params, csv_path, output_path):
    """散布図を作成する関数
       なお、作成時にはeval paramsの全ての組み合わせが作成される

    Args:
        eval_params(list): グラフ特徴量のリスト
        csv_path    (str): 散布図を作成したいcsvfileのpath
        output_path (str): png形式の散布図を保存するディレクトリのpath
                           (例) output_path = "results/2021-01-01_00-00/visualize/scatter_diagram/"
    
    Examples:
        >>> scatter_diagram_visualize('./data/Twitter/twitter.csv')
    """
    dir_name = os.path.splitext(os.path.basename(csv_path))[0]
    df = pd.read_csv(csv_path)
    for param_v in eval_params:
        for param_u in eval_params:
            if re.search('centrality', param_v) or re.search('centrality', param_u) or param_v == param_u:
                continue
            fig = plt.figure()
            x_data = df[param_v]
            y_data = df[param_u]
            sns.jointplot(x=x_data,y=y_data,data=df)
            plt.savefig(output_path + dir_name + '/' + param_v + '_' + param_u + '.png')
            fig.clf()
            plt.close('all')

def histogram_visualize(eval_params, csv_path, output_path):
    """ヒストグラムを作成する関数

    Args:
        eval_params(list): グラフ特徴量のリスト
        csv_path    (str): ヒストグラムを作成したいcsvファイルのパス
        output_path (str): png形式のヒストグラムを保存するディレクトリのパス
                           (例) output_path = "results/2021-01-01_00-00/visualize/histogram/"
    """
    dir_name = os.path.splitext(os.path.basename(csv_path))[0]
    df = pd.read_csv(csv_path)
    for param in eval_params:
        fig = plt.figure()
        if re.search('centrality', param):
            # 全グラフのノードのパラメータを１つのリストにまとめる
            # 原因はわからないがなぜかstrで保存されてしまうのでdictに再変換:ast.literal_eval(graph_centrality)
            total_param = []
            for graph_centrality in df[param]:
                for centrality in ast.literal_eval(graph_centrality).values():
                    total_param.append(centrality)
            sns.histplot(total_param, kde=False)
        else:
            sns.kdeplot(df[param])
        plt.savefig(output_path + dir_name + '/' + param + '.png')
        plt.clf()
        plt.close('all')

def concat_scatter_diagram_visualize(eval_params, csv_paths, output_path):
    """散布図を結合する関数

    Args:
        eval_params(list): グラフ特徴量のリスト
        csv_paths    (str): ヒストグラムを作成したいcsvファイルのパス
        output_path (str): png形式のヒストグラムを保存するディレクトリのパス
                           (例) output_path = "results/2021-01-01_00-00/visualize/histogram/"
    """
    df = utils.concat_csv(csv_paths)
    if 'Diameter' in df.columns:
        df['Diameter'] = df['Diameter'].astype('int64')
    for param_v in eval_params:
        for param_u in eval_params:
            if re.search('centrality', param_v) or re.search('centrality', param_u) or param_v == param_u:
                continue
            fig = plt.figure()
            sns.jointplot(x=df[param_v],y=df[param_u],data=df,hue='type')
            plt.savefig(output_path + param_v + '_' + param_u + '.png')
            fig.clf()
            plt.close('all')

def concat_histogram_visualize(eval_params, csv_paths, output_path,):
    """複数のデータを結合したヒストグラムを作成する関数

    Args:
        eval_params (list): グラフ特徴量のリスト
        csv_paths   (list): ヒストグラムを作成するcsvファイルパスのリスト
        output_path (str) : png形式の結合ヒストグラムを保存するディレクトリのパス
                            (例) output_path = "results/2021-01-01_00-00/visualize/"
    """
    
    if len(csv_paths) == 4:
        color_list = ['orange','green','red','blue']
    elif len(csv_paths) == 5:
        color_list = ['orange','green','red','blue', 'purple']
        
    sns.color_palette("hls",24)
    for param in eval_params:
        fig = plt.figure()
        for path, color in zip(csv_paths, color_list):
            df = pd.read_csv(path)
            label_name = re.sub('.csv', '', path.split("/")[-1])
            sns.kdeplot(df[param],label=label_name, color=color)

        plt.legend(frameon=True)
        plt.savefig(output_path + param + '.png')
        plt.savefig(output_path + param + '.pdf')
        plt.clf()
        plt.close('all')

def pair_plot(eval_params, csv_paths, output_path):
    """Pair plotを作成する関数

    Args:
        eval_params(list): グラフ特徴量のリスト
        csv_paths    (str): ヒストグラムを作成したいcsvファイルのパス
        output_path (str): png形式のヒストグラムを保存するディレクトリのパス
                           (例) output_path = "results/2021-01-01_00-00/visualize/histogram/"
    """
    fig = plt.figure()
    if len(csv_paths) > 1:
        df = utils.concat_csv(csv_paths)
    else:
        df = pd.read_csv(csv_paths[0])
    df = df.reindex(columns=eval_params + ['type'])
    markers = ["o", "s", "D", "X", "*"][0:df['type'].nunique()]  # 必要なマーカー数だけ取り出す
    sns.pairplot(df,hue='type',markers=markers, plot_kws=dict(alpha=0.25))
    plt.savefig(output_path + 'pair_plot.pdf')
    plt.savefig(output_path + 'pair_plot.png', dpi=300)
    fig.clf()
    plt.close('all')

def graph_visualize(graphs, file_name_list, output_path, sampling_num=10, with_colors = False, positions = None):
    """グラフを可視化する関数

    Args:
        graphs          (list): グラフオブジェクトのリスト
        file_name_list   (str): 保存ファイル名
        output_path      (str): 出力先ディレクトリ
                                 e.g. output_path = "result/20220329_171445/visualize/graph_structure/"
        with_colors     (bool): エッジに色を付けるかどうか
        positions        (dict): ノードの位置関係を示すdict
    """
    for i, graph in enumerate(graphs):
        # print(graph.edges())
        if with_colors:
            edges = graph.edges()
            colors = [graph[u][v]['color'] for u,v in edges]
        plt.figure(figsize=(15, 15))
        plt.gca().spines['right'].set_visible(True)
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)  # 余白
        if positions == None:
            pos = nx.spring_layout(graph)
        else:
            pos = positions[i]
        if with_colors:
            nx.draw_networkx(graph, pos, edge_color=colors, node_size=600, width=5)
        else:
            nx.draw_networkx(graph, pos, node_size=600, font_size=18, width=5)
        plt.savefig(output_path + file_name_list[i] + '.png')
        plt.clf()
        plt.close('all')

def dfs_to_animation(dfs_code, end_value_list, position, output_path, edge_num = None, sampled_rank = None):
    """サンプリングの記録をgifとして可視化する関数

    Args:
        dfs_code        ([np.array]): [(sequence,5)のnp.array]
        end_value_list      ([list]): [終了コード[5]]
        position              (dict): ノードの配置を示すdict
        output_path            (str): 出力先ディレクトリ
        edge_num                 int: edge数
        sampled_rank    ([np.array]): [サンプリングで何番目の確率が選ばれたかを格納するnp.array]
    """
    frames = len(dfs_code)

    color_active = 'violet'
    color_non_active = 'black'
    G = nx.Graph()

    # 描画の設定
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    # フレームの軸を固定
    array_pos = np.array(list(position.values()))
    max_xy, min_xy = np.max(array_pos, axis=0), np.min(array_pos, axis=0)
    range_x, range_y = max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]
    xlim = [min_xy[0] - range_x*0.05, max_xy[0] + range_x*0.05]
    ylim = [min_xy[1] - range_y*0.05, max_xy[1] + range_y*0.05]
    close_flag = False

    # 描画の初期関数
    def init_func():
        pass
    
    
    def update(frame):
        """描画のアップデート関数
        Args:
            i (int): 現在のフレーム 
        """
        global close_flag
        if frame == 0: close_flag = False
        # 終了判定
        for index in range(len(dfs_code[frame])):
            if edge_num is not None:
                if len(G.edges) >= edge_num:
                    plt.close(fig)
                    close_flag = True
                elif dfs_code[frame][index] == end_value_list[index]-1:
                    plt.close(fig)
                    close_flag = True
            else:
                # 長さ自体はend_value_listの値だが実際の値は0から始まっているため-1する
                if dfs_code[frame][index] == end_value_list[index]-1:
                    plt.close(fig)
                    close_flag = True

        if not close_flag:
            # ノードとエッジの色をリセット
            if frame != 0:
                for node in G.nodes():
                    G.nodes(data=True)[node]['color'] = color_non_active
                for node0, node1 in G.edges():
                    if sampled_rank is not None:
                        G[node0][node1]['color'] = "black" if ((sampled_rank[0][frame] == 0) and (sampled_rank[1][frame] == 0)) else "red"
                    else:
                        G[node0][node1]['color'] = color_non_active

            tuples = []
            tuples = dfs_code[frame]
            G.add_edge(tuples[0], tuples[1], color=color_active)

            edge_color = [d['color'] for _, _, d in G.edges(data=True)]

            ax.cla()
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
            ax.axis('off')
            fig.suptitle("#{}".format(frame+1),fontsize=30)
            # print(G.nodes())
            nx.draw_networkx(G, position, edge_color=edge_color, ax=ax, node_size=600, font_size=18, width=5)
        

    anim = animation.FuncAnimation(fig, update, frames=frames, init_func=init_func, interval=500)
    anim.save(output_path + ".gif", writer='pillow')
    # anim.save(output_path + ".mp4", writer='ffmpeg')
    return G

def graph_subplot(title, subtitle, result_dir, delete=True, layout=None):

    random.seed(1)

    # graphを画像に変換
    graphs_list = glob.glob(f'./result/{result_dir}/eval/*')
    values = []
    graphs_list = sorted(graphs_list)
    for i in graphs_list:
        value = str(re.search(r'[0-9]+\.[0-9]+', i).group())
        values.append(value)

    if delete:
        try:
            shutil.rmtree(f'./result/{result_dir}/visualize/graph_structure/')
        except:
            pass
    utils.make_dir([f'./result/{result_dir}/visualize/graph_structure/'])
    for i, _ in enumerate(graphs_list):
        utils.make_dir([f'./result/{result_dir}/visualize/graph_structure/{subtitle}_{values[i]}'])

    for index, graph_file in enumerate(graphs_list):
        with open(graph_file, "rb") as f:
            graphs = joblib.load(f)
        if len(graphs) == 3:
            graph = graphs
        else:
            graph = random.sample(graphs, k=3)
        if layout == 'circle':
            positions = [nx.circular_layout(g) for g in graph]
        else:
            positions = None
        base_file_name = f'graph_{subtitle}_{values[index]}_'
        graph_visualize(graph, [f'{base_file_name}{index+1}',f'{base_file_name}{index+2}',f'{base_file_name}{index+3}'],\
            f'./result/{result_dir}/visualize/graph_structure/{subtitle}_{values[index]}/', positions=positions)

    # 画像をcv2型に変換
    graphs_paths = []
    for index in range(len(graphs_list)):
        path = f'./result/{result_dir}/visualize/graph_structure/{subtitle}_{values[index]}/*'
        graphs_path = sorted(glob.glob(path))
        graphs_paths.append(graphs_path)

    graphs_paths = np.array(graphs_paths).T.ravel().tolist()
    
    imgs = []
    for path in graphs_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #pyplotで表示するために色変換
        imgs.append(img)

    row = 3  # 行数
    col = len(values) # 列数
    fig, ax = plt.subplots(nrows=row, ncols=col,figsize=(2.1 / row * col, 2.1), dpi=600)
    fig.subplots_adjust(left=0.04, right=0.95, top=0.87, bottom=0.05, wspace=0.0, hspace=0.0)  # 余白とサブプロット間の間隔
    fig.suptitle(title, fontsize=6, color='black')
    for i, img in enumerate(imgs):
        _r = i//col
        _c = i % col
        if col > 1:
            ax_rc = ax[_r, _c]
        else:
            ax_rc = ax[_r]  # カラムが1になる場合だけ特別処理
        if i < col != 1:
            ax_rc.set_title(f'{subtitle}_{values[i]}', fontsize=5, color='black')
        ax_rc.axes.xaxis.set_visible(False)  # X軸を非表示に
        ax_rc.axes.yaxis.set_visible(False)  # Y軸を非表示に
        ax_rc.spines['right'].set_visible(False)  # 枠線をなくす
        ax_rc.spines['top'].set_visible(False)  # 枠線をなくす
        ax_rc.spines['left'].set_visible(False)  # 枠線をなくす
        ax_rc.spines['bottom'].set_visible(False)  # 枠線をなくす
        ax_rc.imshow(img, cmap='Greys')  # 画像を表示

    plt.savefig(f'./result/{result_dir}/visualize/graph_structure/{title}.png')


if __name__ == '__main__':
    graph_subplot('WS', 'p', 'ws_variation', layout='circle')
