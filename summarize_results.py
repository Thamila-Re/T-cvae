'''
複数のresultのvisualizeをまとめるモジュール.
現状は2nd_rwとBFS, DFSの結果のみ対応.
'''
import glob
import shutil
import argparse
import os
import csv
import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
import numpy as np

from config import common_args, Parameters
from utils import setup_params, make_dir, weighted_mean_std

FONT_SIZE_CELL = 9  # ヒートマップのセルのフォントサイズ
FONT_SIZE_LABEL = 11    # ヒートマップのラベルのフォントサイズ
CONDITION_ABBREVIATION = {'Average path length': 'APL', 'Clustering coefficient': 'CLU', 'Average degree': 'AVD', 'Modularity': 'MOD', 'Power-law exponent': 'POW'} # 特徴量の略称

def summarize_visualize(params: Parameters, mode: str):
    """複数のresultのvisualizeをまとめる関数.

    Args:
		params (config.Parameters): configのグローバル変数のset
        mode (str): まとめる結果の種類
    """
    all_results = [
      dirname for dirname in glob.iglob(params.args['summarize_dir'] + '/*/')
      if 'csv' not in dirname and os.path.isdir(dirname)
    ]
    condition_name = params.condition_params[0]
    condition_values= params.condition_values[condition_name]
    datas_list_all_seed = list()
    for result in sorted(all_results):
        data_dict = dict()
        with open(os.path.join(result,'parameters.json')) as f:
            loaded_params = json.load(f)
        with open(os.path.join(result,'parameters_summarize.json')) as f:
            additional_params = json.load(f)
        if mode == '2nd_rw':
            data_dict['p'] = loaded_params['rw_p']
            data_dict['q'] = loaded_params['rw_q']
        data_dict['train_seed'] = loaded_params['seed']
        data_dict['num_graphs'] = additional_params['number_of_generated_samples']
        
        if params.use_unconnected_graphs:
            vis_dirs = [
                dirname for dirname in glob.iglob(os.path.join(result, 'visualize_*'))
                if 'unconnected' in dirname
            ]
        else:
            vis_dirs = [
                dirname for dirname in glob.iglob(os.path.join(result, 'visualize_*'))
                if 'unconnected' not in dirname
            ]
        for vis_dir in vis_dirs:
            l2_norm_paths = [
                filename for filename in glob.iglob(os.path.join(vis_dir, 'l2_norm', '*'))
                if 'origin' not in filename
            ]
            if len(l2_norm_paths)==0:
                continue
            if params.use_unconnected_graphs:
                # unconnected_ratioの読み込み
                unconnected_ratio_path = os.path.join(vis_dir, 'unconnected_ratio', 'unconnected_ratio.csv')
                unconnected_ratios = list()
                with open(unconnected_ratio_path) as f:
                    csv_reader = csv.reader(f)
                    next(csv_reader)
                    for row in csv_reader:
                        unconnected_ratios.append(float(row[1]))
            for i_cond, l2_path in enumerate(sorted(l2_norm_paths)):
                # l2_normの読み込み
                with open(l2_path) as f:
                    csv_reader = csv.reader(f)
                    next(csv_reader)
                    data_dict['rmse_mean'] = float(next(csv_reader)[i_cond+1])
                if params.use_unconnected_graphs:
                    data_dict['unconnected_ratio'] = unconnected_ratios[i_cond]
                data_dict[condition_name] = condition_values[i_cond]
                datas_list_all_seed.append(data_dict.copy())
    df_total = pd.DataFrame(datas_list_all_seed)

    csv_dir_name = 'csv.unconnected' if params.use_unconnected_graphs else 'csv.connected'
    csv_dir_path = os.path.join(params.args['summarize_dir'], csv_dir_name)
    if os.path.isdir(csv_dir_path):
        shutil.rmtree(csv_dir_path)
    make_dir([csv_dir_path])

    group_list = list()
    if mode == '2nd_rw':
        group_list = [condition_name, 'p', 'q']
    else:
        group_list = [condition_name]

    df_wmean = pd.DataFrame()
    gp = df_total.groupby(group_list, as_index=False)
    datas_list_wmean = list()
    for idx,df_g in gp:
        data_dict_wmean = dict()
        w_mean_l2, w_std_l2 = weighted_mean_std(df_g['rmse_mean'], df_g['num_graphs'])
        if params.use_unconnected_graphs:
            w_mean_unconnected, w_std_unconnected = weighted_mean_std(df_g['unconnected_ratio'], df_g['num_graphs'])
        if type(idx) == tuple:
            data_dict_wmean[condition_name] = idx[0]
        else:
            data_dict_wmean[condition_name] = idx
        data_dict_wmean['rmse_mean'] = w_mean_l2
        data_dict_wmean['rmse_std'] = w_std_l2
        if params.use_unconnected_graphs:
            data_dict_wmean['unconnected_ratio_mean'] = w_mean_unconnected
            data_dict_wmean['unconnected_ratio_std'] = w_std_unconnected
        data_dict_wmean['num_graphs'] = df_g['num_graphs'].sum()
        if mode=='2nd_rw':
            data_dict_wmean['p'] = idx[1]
            data_dict_wmean['q'] = idx[2]
        datas_list_wmean.append(data_dict_wmean)
    df_wmean = pd.DataFrame(datas_list_wmean)
    if mode != '2nd_rw': 
        if params.use_unconnected_graphs:
            df_wmean.drop(columns=['unconnected_ratio_mean', 'unconnected_ratio_std']).to_csv(os.path.join(csv_dir_path, 'l2_' + mode + '.csv'), index = False)
            df_wmean.drop(columns=['rmse_mean', 'rmse_std']).to_csv(os.path.join(csv_dir_path, 'unconnected_' + mode + '.csv'), index = False)
        else:
            df_wmean.to_csv(os.path.join(csv_dir_path, 'l2_' + mode + '.csv'), index = False)
    else:
        if params.use_unconnected_graphs:
            metrics = ['rmse', 'unconnected_ratio']
            connected_text = 'unconnected'
        else:
            metrics = ['rmse']
            connected_text = 'connected'
        for metric in metrics:
            df = pd.DataFrame()
            if metric == 'rmse':
                df = df_wmean.drop(columns=['unconnected_ratio_mean', 'unconnected_ratio_std'])
            else:
                df = df_wmean.drop(columns=['rmse_mean', 'rmse_std'])
            pivot = pd.pivot_table(df, [metric+'_mean', metric+'_std'], [condition_name, 'p'], [condition_name, 'q'])
            pivot.to_csv(os.path.join(csv_dir_path, metric + '_all.csv'))
            if os.path.join(params.args['heatmap_dir'], metric) is not None:
                make_dir([params.args['heatmap_dir'], os.path.join(params.args['heatmap_dir'], metric)])
            for c in condition_values:
                pivot_each = pd.pivot_table(df[df[condition_name]==c], [metric+'_mean', metric+'_std'], [condition_name, 'p'], [condition_name, 'q'])
                pivot_each.to_csv(os.path.join(csv_dir_path, metric + '_'+condition_name+str(c)+'.csv'))
                if params.args['heatmap_dir'] is not None:
                    # ヒートマップの作成.
                    dfs_dir_path = Path(csv_dir_path).parents[1].joinpath('dfs', 'csv.' + connected_text)
                    bfs_dir_path = Path(csv_dir_path).parents[1].joinpath('bfs', 'csv.' + connected_text)
                    df_dict_dfs_bfs = load_dfs_bfs_summarize(
                        {'dfs': dfs_dir_path, 'bfs': bfs_dir_path}, 
                        metric=metric,
                        condition_name=condition_name, 
                        condition_value=c
                    )
                    if connected_text == 'unconnected':
                        # unconnectedの場合はファイル名にconnectedを含めない
                        heatmap_file_name = metric + '_d' +str(int(params.split_size['train'] * params.input_original_graph_num)) + '_' + CONDITION_ABBREVIATION[condition_name] + '{:.2f}'.format(c) + '.pdf'
                    else:
                        heatmap_file_name = metric + '_d' +str(int(params.split_size['train'] * params.input_original_graph_num)) + '_' + CONDITION_ABBREVIATION[condition_name] + '{:.2f}'.format(c) + '_connected.pdf'
                    save_heatmap(
                        df[df[condition_name]==c].copy().drop(columns=[condition_name, 'num_graphs']), 
                        os.path.join(params.args['heatmap_dir'], metric, heatmap_file_name), 
                        metric=metric, 
                        df_dict=df_dict_dfs_bfs,
                        round_num=3 if metric == 'rmse' else 2
                    )

def load_dfs_bfs_summarize(dir_path_dict: dict, metric: str, condition_name: str=None, condition_value: float=None):
    '''dfsとbfsの結果を読み込む関数.

    Args:
        dir_path_dict (dict): dfsとbfsの結果のディレクトリパス
        metric (str): 評価指標
        condition_name (str): 特徴量の名前
        condition_value (float): 特徴量の値
    Returns:
        dict: dfsとbfsの結果のDataFrameの辞書
    '''
    search = ['dfs', 'bfs']
    df_dict = dict()
    for s in search:
        if not os.path.isdir(dir_path_dict[s]):
            return None, None
        if metric == 'rmse':
            with open(os.path.join(dir_path_dict[s], 'l2_' + s + '.csv')) as f:
                df = pd.read_csv(f)
                df_dict[s] = df[df[condition_name]==condition_value].copy().drop(columns=[condition_name, 'num_graphs'])
                df_dict[s].rename(columns={'l2norm_mean': 'rmse_mean', 'l2norm_std': 'rmse_std'}, inplace=True)
                df_dict[s]['search'] = s
        elif metric == 'unconnected_ratio':
            with open(os.path.join(dir_path_dict[s], 'unconnected_' + s + '.csv')) as f:
                df = pd.read_csv(f)
                df_dict[s] = df[df[condition_name]==condition_value].copy().drop(columns=[condition_name, 'num_graphs'])
                df_dict[s]['search'] = s
    return df_dict

def save_heatmap(df, save_path, metric='rmse', df_dict=None, round_num=3):
    '''ヒートマップを保存する関数.

    Args:
        df (pd.DataFrame): ヒートマップのデータ
        save_path (str): 保存先のパス
        metric (str): 評価指標
        df_dict (dict): dfsとbfsの結果のDataFrameの辞書
        round_num (int): 丸める桁数
    '''
    df_annotation = df.copy()
    round_map = lambda x: '{x:.{round_num}f}'.format(x=x, round_num=round_num)
    df_annotation['annotation'] = df[metric +'_mean'].map(round_map) + '±' + df[metric + '_std'].map(round_map)
    pivot_annotation = pd.pivot_table(df_annotation.drop(columns=[metric +'_mean', metric +'_std']), index='p', columns='q', values='annotation', aggfunc=lambda x: ''.join(str(v) for v in x))
    pivot = pd.pivot_table(df.drop(columns=metric +'_std'), index='p', columns='q', values=metric +'_mean')
    min_total = min(df[metric +'_mean'].min(), df_dict['dfs'][metric +'_mean'].min(), df_dict['bfs'][metric +'_mean'].min())
    max_total = max(df[metric +'_mean'].max(), df_dict['dfs'][metric +'_mean'].max(), df_dict['bfs'][metric +'_mean'].max())
    df_concat = pd.concat([df_annotation, df_dict['dfs'], df_dict['bfs']], axis=0)
    df_concat.drop(columns=[metric + '_std'], inplace=True)
    df_concat['search'].fillna('2nd_rw', inplace=True)
    vals = df_concat[metric +'_mean'].values
    cm = plt.get_cmap('coolwarm')
    norm = Normalize(vmin=min_total, vmax=max_total)
    color = [cm(norm(val)) for val in vals]
    dfs_color = color[-2][:]
    bfs_color = color[-1][:]

    min_row = df_concat.loc[df_concat[metric +'_mean'] == min_total]

    # sns.set_theme(font_scale=1.4)
    fig = plt.figure(figsize=(6.4, 4.8))
    ax_2rw = fig.add_subplot(3, 1, (1, 2))
    ax_dfs = fig.add_subplot(3, 2, 5)
    ax_dfs.axis('off')
    ax_bfs = fig.add_subplot(3, 2, 6)
    ax_bfs.axis('off')

    if min_row['search'].values[0] == '2nd_rw':
        mask_pivot = pivot == min_total
    else:
        mask_pivot = np.empty_like(pivot, dtype=bool)
        mask_pivot[:,:] = False

    sns.heatmap(pivot, annot=pivot_annotation, fmt='', cmap='coolwarm', ax=ax_2rw, vmax=max_total, vmin=min_total, cbar_kws = dict(location="top"), annot_kws={'fontsize': FONT_SIZE_CELL}, linewidth=.5, linecolor='black', mask=mask_pivot)
    sns.heatmap(pivot, annot=pivot_annotation, fmt='', cmap='coolwarm', ax=ax_2rw, vmax=max_total, vmin=min_total, cbar=False, annot_kws={'fontsize': FONT_SIZE_CELL, 'weight': 'bold'}, linewidth=.5, linecolor='black', mask=~mask_pivot)
    ax_2rw.text(0.6, 6.60, '2nd-order random walk-based conversion',fontsize=FONT_SIZE_LABEL, weight='bold')
    
    dfs_values = df_dict['dfs'][metric +'_mean'].map(round_map) + '±' + df_dict['dfs'][metric + '_std'].map(round_map)
    dfs_values = dfs_values.values[0]
    if float(norm(df_dict['dfs'][metric +'_mean'].values)) < 0.3 or float(norm(df_dict['dfs'][metric +'_mean'].values)) > 0.7:
        dfs_font_color = 'white'
    else:
        dfs_font_color = 'black'
    bfs_values = df_dict['bfs'][metric +'_mean'].map(round_map) + '±' + df_dict['bfs'][metric + '_std'].map(round_map)
    bfs_values = bfs_values.values[0]
    if float(norm(df_dict['bfs'][metric +'_mean'].values)) < 0.3 or float(norm(df_dict['bfs'][metric +'_mean'].values)) > 0.7:
        bfs_font_color = 'white'
    else:
        bfs_font_color = 'black'

    dfs_table = ax_dfs.table(cellText=[[dfs_values]], loc='top', cellLoc='center', cellColours=[[dfs_color]], bbox=[0.22, -0.06, 0.42, 0.42])
    ax_dfs.text(-0.05, -0.21, 'DFS code-based conversion', fontsize=FONT_SIZE_LABEL, weight='bold')
    bfs_table = ax_bfs.table(cellText=[[bfs_values]], loc='top', cellLoc='center', cellColours=[[bfs_color]], bbox=[0.22, -0.06, 0.42, 0.42])
    ax_bfs.text(-0.05, -0.21, 'BFS code-based conversion', fontsize=FONT_SIZE_LABEL, weight='bold')

    dfs_table.auto_set_font_size(False)
    if min_row['search'].values[0] == 'dfs':
        for (_, _), cell in dfs_table.get_celld().items():
            if cell.get_text().get_text() == dfs_values:
                cell.set_text_props(fontproperties=FontProperties(weight='bold'))
    elif min_row['search'].values[0] == 'bfs':
        for (_, _), cell in bfs_table.get_celld().items():
            if cell.get_text().get_text() == bfs_values:
                cell.set_text_props(fontproperties=FontProperties(weight='bold'))
    dfs_table.set_fontsize(FONT_SIZE_CELL)
    # dfs_table.scale(0.45, 4.2)
    bfs_table.auto_set_font_size(False)
    # bfs_table.scale(0.45, 4.2)
    bfs_table.set_fontsize(FONT_SIZE_CELL)
    dfs_table[0, 0].get_text().set_color(dfs_font_color)
    bfs_table[0, 0].get_text().set_color(bfs_font_color)
    
    c_bar = ax_2rw.collections[0].colorbar
    c_bar.set_ticks([min_total, (min_total+max_total)/2, max_total])
    c_bar.set_ticklabels(['Low', 'Medium', 'High'])

    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = common_args(parser)
    args = parser.parse_args()
    params = Parameters(**setup_params(vars(args), args.parameters))  # args，run_date，git_revisionなどを追加した辞書を取得
    summarize_visualize(params, mode=params.search)