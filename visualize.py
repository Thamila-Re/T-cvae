"""
可視化を実行するするためのインターフェースがまとめられたモジュール.
"""
from ctypes import util
import enum
import glob
from turtle import position
import joblib
import shutil
import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
from numpy import require, sort
from sklearn.manifold import TSNE
from sympy import im
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import networkx as nx

from config import common_args, Parameters
from utils import dump_params, setup_params
import json
from logging import Logger, getLogger
from utils import set_logging
import utils
from graph_process import convert_to_sequence_code, graph_statistic
from graph_process import complex_networks
import bi
from models import cvae
from graph_process import graph_utils
from graph_process import graph_sampling

logger = getLogger(__name__)

def graph_plot(params: Parameters, visualize_dir_name: str = 'visualize') -> None:
	"""グラフ特徴量をplotするための関数.

	Args:
		params (config.Parameters): configのグローバル変数のset
	"""
	calc_graph_statistic = graph_statistic.GraphStatistic(params)

	# eval.py で生成されたグラフの特徴量をcsvへ書き出し
	logger.info("Start : Graph Plot")
	if params.args['eval_graphs']:
		logger.info("params.args['eval_graphs']をチェック")
		if params.args['eval_graphs'][-1] != "/":
			logger.warning("パスの末尾に / を付けてください.")
			exit()
		logger.info("生成されたグラフを読み込み.")
		logger.info("Graph path : " + params.args['eval_graphs'])
		visualize_dir = str(Path(params.args["eval_graphs"]).parent) + "/" + visualize_dir_name + "/"
		if params.args['out_csv']: # out_csvはそこそこ時間がかかるので、デバッグ時に動作確認だけしたいときだけFalseにする.
			logger.info("Start : Output Condition of Generated Graphs to CSV") 
			if not os.path.isdir(visualize_dir + "csv"):
				utils.make_dir([visualize_dir + "csv"])
			graph_files = glob.glob(params.args['eval_graphs'] + r"*.pkl")     # 生成されたグラフのパスのリスト
			if params.use_unconnected_graphs:
				logger.info("Unconnected Graphs are used.")
			else:
				logger.info("Unconnected Graphs are not used.")
			unconnected_ratio_dict = dict()
			for graph_file in graph_files:
				with open(graph_file, "rb") as f:
					loaded_graphs = joblib.load(f)
					# グラフオブジェクトから特徴量を計算し, csvファイルへ出力.
					logger.info(f"Total Graphs in {f.name} : {len(loaded_graphs)}")
					graphs_to_csv = list()
					unconnected_count = 0
					for graph in loaded_graphs:
						if nx.is_connected(graph):
							graphs_to_csv.append(graph)
						else:
							unconnected_count += 1
							if params.use_unconnected_graphs:
								graphs_to_csv.append(calc_graph_statistic.largest_component_subgraph(graph))
					logger.info(f"Unconnected Graphs in {f.name} : {unconnected_count}")
					if params.args["unconnected_ratio"]:
						unconnected_ratio_dict.update({f.name.split("/")[-1] : (unconnected_count / len(loaded_graphs) * 100, unconnected_count)})
					calc_graph_statistic.graph2csv(graphs_to_csv, csv_dir=visualize_dir+"csv/", file_name=graph_file.split("/")[-1], eval_params=params.eval_params)
			if params.args["unconnected_ratio"]:
				unconnected_ratio_dir = visualize_dir + "unconnected_ratio/"
				if os.path.isdir(unconnected_ratio_dir):
					shutil.rmtree(unconnected_ratio_dir)
				utils.make_dir([unconnected_ratio_dir])
				calc_graph_statistic.unconnected_ratio(unconnected_ratio_dict, unconnected_ratio_dir)
			logger.info("Finish : Output Condition of Generated Graphs to CSV")
	else:
		# 新規に結果出力用ディレクトリを作成
		result_dir = params.result_dir
		required_dirs = [result_dir, result_dir+"/train", result_dir+"/eval", result_dir+"/" + visualize_dir_name, result_dir+"/"+ visualize_dir_name+ "/csv"]
		utils.make_dir(required_dirs)
		visualize_dir = result_dir + "/" + visualize_dir_name + "/"

	if params.args['out_csv']: # out_csvはそこそこ時間がかかるので、デバッグ時に動作確認だけしたいときだけFalseにする.
		# データセットをcsvへ書き出し
		logger.info("Start : Output Condition of Dataset Graphs to CSV")
		complex_network = complex_networks.ComplexNetworks(params=params)
		datasets = complex_network.pickup_origin_data(params.pickup_num)
		for (detail_name, detail) in params.visualize_detail.items():
			# calc_graph_statistic.graph2csv(datasets[detail_name][0], csv_dir=visualize_dir+"csv/", file_name=detail_name, eval_params=params.eval_params)
			calc_graph_statistic.graph2csv(datasets, csv_dir=visualize_dir+"csv/", file_name=params.args['dataset']+"_origin", eval_params=params.eval_params)
		logger.debug("Finish : Output Condition of Dataset Graphs to CSV")
		
	csv_paths = glob.glob(visualize_dir + "csv/*")
	csv_paths = sorted(csv_paths)
	# 各グラフ特徴量の平均値を計算する
	if params.args['average_param']:
		logger.info("Start : Calculate Average of Graphs")
		if os.path.isdir(visualize_dir + "average_param/"):
			shutil.rmtree(visualize_dir + "average_param")
		average_dir = visualize_dir + "average_param/"
		if os.path.isdir(visualize_dir + "percentile_param/"):
			shutil.rmtree(visualize_dir + "percentile_param")
		percentile_dir = visualize_dir + "percentile_param/"
		required_dirs = [average_dir, percentile_dir]
		utils.make_dir(required_dirs)
		for path in csv_paths:
			calc_graph_statistic.get_average_params(path, average_dir)
			calc_graph_statistic.get_percentile_params(path, percentile_dir)
		logger.debug("Finish : Calculate Average of Graphs")

	# 各グラフ特徴量のL2 normを計算する
	if params.args['l2_norm']:
		logger.info("Start : Calculate L2 Norm of Graphs")
		if os.path.isdir(visualize_dir + "l2_norm/"):
			shutil.rmtree(visualize_dir + "l2_norm")
		l2_norm_dir = visualize_dir + "l2_norm/"
		required_dirs = [l2_norm_dir]
		utils.make_dir(required_dirs)
		## L2 normを計算し、csvで出力
		for path in csv_paths:
			calc_graph_statistic.get_l2_norm_params(
				condition_params=params.condition_params,
				condition_values=params.condition_values,
				csv_path=path, save_dir=l2_norm_dir)
		logger.debug("Finish : Calculate L2 Norm of Graphs")

	# 散布図を作成
	if params.args['scatter_diagram']:
		logger.info("Start : Create Scatter Diagram")
		if os.path.isdir(visualize_dir + "scatter_diagram/"):
			shutil.rmtree(visualize_dir + "scatter_diagram")
		## csvのファイルパスからdir名を持ってくる
		dir_names = [os.path.splitext(os.path.basename(csv_path))[0] for csv_path in csv_paths]
		## dir名からdirを生成
		required_dirs = [visualize_dir + "scatter_diagram"] + [visualize_dir + "scatter_diagram/" + dir_name for dir_name in dir_names]
		utils.make_dir(required_dirs)
		for path in csv_paths:
			bi.scatter_diagram_visualize(params.eval_params, path, visualize_dir+"scatter_diagram/")
		logger.debug("Finish : Create Scatter Diagram")

	# ヒストグラムを作成
	if params.args['histogram']:
		logger.info("Start : Create Histogram")
		if os.path.isdir(visualize_dir + "histogram/"):
			shutil.rmtree(visualize_dir + "histogram")
		## csvのファイルパスからdir名を持ってくる
		dir_names = [os.path.splitext(os.path.basename(path))[0] for path in csv_paths]
		## dir名からdirを生成
		required_dirs = [visualize_dir + "histogram"] + [visualize_dir + "histogram/" + dir_name for dir_name in dir_names]
		utils.make_dir(required_dirs)
		for path in csv_paths:
			bi.histogram_visualize(params.eval_params, path, output_path=visualize_dir+"histogram/")
		logger.debug("Finish : Create Histogram")

	# 散布図を結合
	if params.args['concat_scatter_diagram']:
		logger.info("Start : Concat Scatter Diagram")
		if os.path.isdir(visualize_dir + "concat_scatter_diagram/"):
			shutil.rmtree(visualize_dir + "concat_scatter_diagram")
		dir_name = ''
		for index,path in enumerate(csv_paths):
			dir_name += os.path.splitext(os.path.basename(path))[0]
			if index != len(csv_paths)-1:
				dir_name += '&'
		## dir名からdirを生成
		required_dirs = [visualize_dir + "concat_scatter_diagram"] + [visualize_dir + "concat_scatter_diagram/" + dir_name]
		utils.make_dir(required_dirs)
		bi.concat_scatter_diagram_visualize(params.eval_params, csv_paths, visualize_dir+"concat_scatter_diagram/"+dir_name+"/")
		logger.debug("Finish : Concat Scatter Diagram")

	# ヒストグラムを結合
	if params.args['concat_histogram']:
		logger.info("Start : Concat Histogram")
		if os.path.isdir(visualize_dir + "concat_histogram/"):
			shutil.rmtree(visualize_dir + "concat_histogram")
		dir_name = ''
		for index,path in enumerate(csv_paths):
			dir_name += os.path.splitext(os.path.basename(path))[0]
			if index != len(csv_paths)-1:
				dir_name += '&'
		## dir名からdirを生成
		required_dirs = [visualize_dir + "concat_histogram"] + [visualize_dir + "concat_histogram/" + dir_name]
		utils.make_dir(required_dirs)
		bi.concat_histogram_visualize(params.eval_params, csv_paths, visualize_dir+"concat_histogram/"+dir_name+"/")
		logger.debug("Finish : Concat Histogram")

	# pair plotを作成
	if params.args['pair_plot']:
		logger.info("Start : Create Pair Plot")
		if os.path.isdir(visualize_dir + "pair_plot/"):
			shutil.rmtree(visualize_dir + "pair_plot")
		required_dirs = [visualize_dir + "pair_plot"]
		utils.make_dir(required_dirs)
		bi.pair_plot(params.eval_params, csv_paths, visualize_dir+"pair_plot/")
		logger.debug("Finish : Create Pair Plot")

	# GED(Graph Edit Distance)を計算して、csvで出力
	if params.args['ged']:
		logger.info("Start : Calculate GED")
		ged_dir = visualize_dir + "ged/"
		if not params.args['skip_calc_ged']:
			if os.path.isdir(visualize_dir + "ged/"):
				shutil.rmtree(visualize_dir + "ged")
			required_dirs = [ged_dir]
			utils.make_dir(required_dirs)
		## GEDを計算し、csvで出力
		calc_graph_statistic.get_graph_edit_distance(
            save_dir=ged_dir, all_csv=True)
		logger.debug("Finish : Calculate GED")

	if params.args['quality']:
		logger.info("Start : Calculate Uniqueness/Novelty of Generated Graphs")
		quality_dir = visualize_dir + "quality/"
		if os.path.isdir(visualize_dir + "quality/"):
			shutil.rmtree(visualize_dir + "quality")
		required_dirs = [quality_dir]
		utils.make_dir(required_dirs)
		# Uniqunessを計算し、csvで出力
		calc_graph_statistic.get_quality(
			save_dir=quality_dir,
			eval_graphs=params.args['eval_graphs'])
		logger.debug("Finish : Calculate Uniquness")
			
	logger.debug("Finish : Graph Plot")


def graph_visualize(graph_path:str, visualize_dir:str, sampling_num:int=10) -> None:
    """グラフを可視化する関数

    Args:
        graph_path 		(str): グラフデータまでのpathの正規表現
                                e.g. graph_path = "result/20220329_171445/eval/*"
        visualize_dir	(str): graph_pathに対応するvisualizeディレクトリ
        sampling_num (int, optional): サンプリングする数. Defaults to 10.
    """
    result_dir = graph_path.split("/")[0] + "/" + graph_path.split("/")[1] + "/"
    output_path = visualize_dir + "graph_structure/"
    if os.path.isdir(visualize_dir + "graph_structure/"):
        shutil.rmtree(visualize_dir + "graph_structure")
    required_dirs = [visualize_dir + "graph_structure"]
    utils.make_dir(required_dirs)
    # load graphs
    graph_files = glob.glob(graph_path)
    for graph_file in graph_files:
        with open(graph_file, "rb") as f:
            graphs = joblib.load(f)
            # sampling
            numbers = [i for i in range(0, len(graphs), 1)]
            indicies = random.sample(numbers, sampling_num)
            sampled_graphs = [graphs[i] for i in indicies]
            file_name_list = [f"{graph_file.split('/')[-1]}_{i}" for i in indicies]
            # graph visualize
            bi.graph_visualize(sampled_graphs, file_name_list, output_path)
    logger.info("graph visualize complete!")


def dfs_visualize(params: Parameters):
	"""dfsコードからグラフが生成される過程を可視化する関数.

	Args:
		params (config.Parameters): configのグローバル変数のset
	"""
	# 生成したdfsコードからグラフへの変換過程を可視化
	# Deviceはcpuに限定
	device = "cpu"
	# Load preprocessed dataset
	time_size, node_size, edge_size, conditional_size = joblib.load(os.path.join(str(Path(params.args["eval_graphs"]).parent), "dataset", "param"))
	# if params.use_transition_flag:
	# 	edge_size = 3

	# ディレクトリ作成
	visualize_dir = os.path.join(str(Path(params.args["eval_graphs"]).parent), "visualize") + "/"
	if os.path.isdir(visualize_dir + "dfs_history/"):
		shutil.rmtree(visualize_dir + "dfs_history")
	dfs_history_dir = visualize_dir + "dfs_history/"
	required_dirs = [dfs_history_dir, dfs_history_dir + "animations/", dfs_history_dir + "png/", dfs_history_dir + "animations/datasets/",  
	dfs_history_dir + "animations/generated/", dfs_history_dir + "png/datasets/", dfs_history_dir + "png/generated/"]
	utils.make_dir(required_dirs)

	# データセットのグラフをDFSに変換
	complex_network = complex_networks.ComplexNetworks(params=params)
	datasets_dfs_code = list()
	# datasets = complex_network.create_dataset(params.visualize_detail, do_type='visualize')
	if params.args["dataset"] == "twitter":
		datasets = complex_network.pickup_twitter_data(params.dfs_visualize_detail["twitter_pickup"][0])
	elif params.args["dataset"] == "reddit":
		datasets = complex_network.pickup_reddit_data(params.dfs_visualize_detail["reddit_pickup"][0])
    
	for dataset_graph in datasets:
        #covert_graph = graph_process.ConvertToDfsCode(graph)
		if(params.search == "2nd_rw"):
			covert_graph = graph_sampling.GraphSampling(dataset_graph, node_order=params.dfs_mode, search_type=params.search, p=params.rw_p, q=params.rw_q)
		else:
			covert_graph = convert_to_sequence_code.ConvertToSequenceCode(dataset_graph, mode=params.dfs_mode, search_type=params.search)
		tmp = covert_graph.get_seq_code()
        # 一旦tmpにdfscodeを出してからdfscodeにappend
		datasets_dfs_code.append(tmp)
	
	# evalで生成されたDFSコードを読み込む
	if params.args['eval_graphs']:
		logger.info("params.args['eval_graphs'] をチェック.")
		if params.args['eval_graphs'][-1] != "/":
			logger.info("パスの末尾に / を付けてください.")
			exit()
		if os.path.isdir(params.args['eval_graphs'] + "dfs_history/"):
			logger.info("dfsコードとサンプリングの記録を読み込み中.")
			history_path = params.args['eval_graphs'] + "dfs_history/"
			result_all = torch.load(history_path+"dfs_code.pt")
			sampled_rank = torch.load(history_path+"sampled_rank.pt")
			with open(history_path+"condition.txt","r") as f:
				conditional_labels = f.read().splitlines()
			logger.info("読み込み完了.")
		else:
			logger.info("DFS codeが存在しません.")
			exit()

	dataset_output_path = [dfs_history_dir + "png/datasets/", dfs_history_dir + "animations/datasets/"]
	file_name_list = list()

	positions = list()
	for index, dataset_graph in enumerate(datasets):
		positions.append(nx.spring_layout(dataset_graph))
		dataset_file_name = "Data" + "{0:03d}".format(index + 1)
		file_name_list.append(dataset_file_name)

	if params.args["history_vis"] == 'gif' or params.args["history_vis"] == 'both':
		logger.info("Datasetのアニメーションを作成中")
		for index, dataset_dfs in enumerate(datasets_dfs_code):
			# dataset graphs
			bi.dfs_to_animation(dataset_dfs,
					[time_size, time_size, node_size, node_size, edge_size],
					position = positions[index],
					output_path = dataset_output_path[1] + "Data" + "{0:03d}".format(index + 1))
			
		logger.info("アニメーション作成完了")

	if params.args["history_vis"] == 'png' or params.args["history_vis"] == 'both':
		logger.info("Datasetのpngを作成中")
		bi.graph_visualize(datasets, file_name_list, dataset_output_path[0], positions=positions)
		logger.info("png作成完了")

	graphs = list()
	file_name_list.clear()
	positions.clear()
	generated_output_path = [dfs_history_dir + "png/generated/", dfs_history_dir + "animations/generated/"]

	for index, (result, cond_label) in enumerate(zip(result_all, conditional_labels)):
        # generated graphs
		result = [code.unsqueeze(2) for code in result]
		dfs_code = torch.cat(result, dim=2)
		for i, code in enumerate(dfs_code):
			generated_graph = graph_utils.dfs_code_to_graph_obj(
				code.cpu().detach().numpy(),
                [time_size, time_size, node_size, node_size, edge_size],
                edge_num = params.generate_edge_num,
				sampled_rank= sampled_rank[index][i])
			generated_file_name = "_".join(params.condition_params) + "_" + str(cond_label) + "_" + "{0:03d}".format(i + 1)
			file_name_list.append(generated_file_name)
			pos = nx.spring_layout(generated_graph)
			if params.args["history_vis"] == 'gif' or params.args["history_vis"] == 'both':
				# gif化
				logger.info(f"{params.condition_params[0]}={cond_label}のグラフのアニメーションを作成中")
				bi.dfs_to_animation(code.cpu().detach().numpy(),
					[time_size, time_size, node_size, node_size, edge_size],
					position = pos,
					output_path = generated_output_path[1] + generated_file_name,
					sampled_rank = sampled_rank[index][i],
					edge_num=params.generate_edge_num)
				logger.info(f"{params.condition_params[0]}={cond_label}のアニメーション作成完了")
			graphs.append(generated_graph)
			positions.append(pos)
			
		if params.args["history_vis"] == 'png' or params.args["history_vis"] == 'both':
			logger.info(f"{params.condition_params[0]}={cond_label}のグラフのpngを作成中")
			bi.graph_visualize(graphs=graphs, file_name_list=file_name_list, output_path=generated_output_path[0], with_colors=True, positions=positions)
			logger.info("png作成完了")
		graphs.clear()
		file_name_list.clear()
		positions.clear()
	
	logger.info("dfs_visualize complete!")


def get_degree_distribution(params: Parameters):
	"""データセットのグラフと生成したグラフの次数分布を保存する関数。
	Args:
		params (config.Parameters): configのグローバル変数のset
	"""
	visualize_dir = "result/" + params.args['eval_graphs'].split("/")[1] + "/visualize/"
	if os.path.isdir(visualize_dir + "degree_distribution/"):
		shutil.rmtree(visualize_dir + "degree_distribution/")
	degree_dist_dir = visualize_dir + "degree_distribution/"
	required_dirs = [degree_dist_dir, degree_dist_dir + "dataset", degree_dist_dir + "generated"]
	utils.make_dir(required_dirs)

	
	complex_network = complex_networks.ComplexNetworks(params=params)
	if params.args["dataset"] == "twitter":
		datasets = complex_network.pickup_twitter_data(params.dfs_visualize_detail["twitter_pickup"][0])
	elif params.args["dataset"] == "reddit":
		datasets = complex_network.pickup_reddit_data(params.dfs_visualize_detail["reddit_pickup"][0])		
	for index,dataset_graph in enumerate(datasets):
		dataset_degree = [d[1] for d in dataset_graph.degree]
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.hist(dataset_degree, bins=range(0, 40), density=True, edgecolor='black')
		plt.ylim(0.0, 1.0)
		plt.xlim(0, 40)
		plt.savefig(degree_dist_dir + "dataset/" + "Data" + "{0:03d}".format(index + 1)+ ".png")
		plt.close()

	if os.path.isdir(params.args['eval_graphs'] + "dfs_history/"):
		history_path = params.args['eval_graphs'] + "dfs_history/"
		with open(history_path+"condition.txt","r") as f:
			conditional_labels = f.read().splitlines()

	generated_graph_files = glob.glob(params.args['eval_graphs'] + r"*")     # 生成されたグラフのパスのリスト
	generated_graph_lists = list()
	for generated_graph_file in generated_graph_files:
		with open(generated_graph_file, "rb") as f:
			generated_graphs = joblib.load(f)
			generated_graph_lists.append(generated_graphs)

	for index, (generated_graph, cond_label) in enumerate(zip(generated_graph_lists, conditional_labels)):
		for i, generated_g in enumerate(generated_graph):
			if i == 30:
				break
			generated_degree = [d[1] for d in generated_g.degree]
			# generated_degree_norm = [d/nx.number_of_nodes(generated_g) for d in generated_degree]
			fig = plt.figure()
			ax = fig.add_subplot(1,1,1)
			ax.hist(generated_degree, bins=range(0, 40), density=True, edgecolor='black')
			plt.ylim(0.0, 1.0)
			plt.xlim(0, 40)
			plt.savefig(degree_dist_dir + "generated/" + "_".join(params.condition_params) + "_" + str(cond_label) + "_" + "{0:03d}".format(i + 1)+ ".png")
			plt.close()


def visualize_rep(params: Parameters):
    """CVAEモデルの潜在変数をt-sne可視化

    Args:
        params (config.Parameters): グローバル変数のセット
    """
    # 前処理後のdataをload
    valid_dataset = joblib.load("dataset/valid/onehot")
    valid_label = joblib.load("dataset/valid/label")
    valid_conditional = joblib.load("dataset/valid/conditional")
    time_size, node_size, edge_size, conditional_size = joblib.load("dataset/param")
    #
    dfs_size = 2 * time_size + 2 * node_size + edge_size + conditional_size
    dfs_size_list = [time_size, time_size, node_size, node_size, edge_size]
    valid_conditional = torch.cat([valid_conditional for _ in range(valid_dataset.shape[1])], dim=1).unsqueeze(2)
    valid_dataset = torch.cat((valid_dataset, valid_conditional), dim=2)
    # dataloaderを作成
    g = torch.Generator()
    g.manual_seed(params.seed)
    valid_label_args = torch.LongTensor(list(range(valid_dataset.shape[0])))
    valid_dl = DataLoader(
        TensorDataset(valid_label_args, valid_dataset),
        shuffle=False,
        batch_size=params.model_params["batch_size"],
        pin_memory=True,
        generator=g
    )
    # Load model
    model = cvae.CVAE(dfs_size, time_size, node_size, edge_size, conditional_size, params, 'cpu')
    model.load_state_dict(torch.load(params.args["eval_model"], map_location="cpu"))
    model.eval()
    # forward
    for i, (indicies, data) in enumerate(valid_dl, 0):
        mu, sigma, tu, tv, lu, lv, le = model(data)
        z = mu + torch.exp(0.5 * sigma) * torch.randn(sigma.shape)
        colors = ["red", "green", "blue", "orange", "purple", "brown", "fuchsia", "grey", "olive", "lightblue"]
        plt.figure(figsize=(10, 10))
        points = TSNE(n_components=2, random_state=0).fit_transform(z.squeeze(dim=1).detach().numpy())
        for p, l in zip(points, valid_conditional[indicies].detach().numpy()):
            plt.scatter(p[0], p[1], marker="${}$".format(round(l[0][0], 0)), c=colors[round(l[0][0], 0)])
        plt.savefig("tsne.png")
        plt.close()
        break


def visualize_augment_dataset(params: Parameters,
                                visualize_dir: str,
                                graph_file: str = None):
    """拡張データセットの特徴量の分布を可視化する

    Args:
        params (config.Parameters): グローバル変数のセット
        visualize_dir        (str): 可視化結果を格納するdir
                                    e.g. "augment_dataset/aug_dataset_50%_edges_visualize/"
        graph_file (str, optional): graph objが保存されているファイルへのpath. Defaults to None.
                                    e.g. 'dataset/augment_data_based_twitter_edge_50.joblib'
    """
    calc_graph_statistic = graph_statistic.GraphStatistic()

    csv_paths = glob.glob(visualize_dir + "csv/*")
    csv_paths = sorted(csv_paths)

    # 各グラフ特徴量の平均値を計算する
    if os.path.isdir(visualize_dir + "average_param/"):
        shutil.rmtree(visualize_dir + "average_param")
    average_dir = visualize_dir + "average_param/"
    if os.path.isdir(visualize_dir + "percentile_param/"):
        shutil.rmtree(visualize_dir + "percentile_param")
    percentile_dir = visualize_dir + "percentile_param/"
    required_dirs = [average_dir, percentile_dir]
    utils.make_dir(required_dirs)
    for path in csv_paths:
        calc_graph_statistic.get_average_params(path, average_dir)
        calc_graph_statistic.get_percentile_params(path, percentile_dir)

    # 散布図を作成
    if os.path.isdir(visualize_dir + "scatter_diagram/"):
        shutil.rmtree(visualize_dir + "scatter_diagram")
    ## csvのファイルパスからdir名を持ってくる
    dir_names = [os.path.splitext(os.path.basename(csv_path))[0] for csv_path in csv_paths]
    ## dir名からdirを生成
    required_dirs = [visualize_dir + "scatter_diagram"] + [visualize_dir + "scatter_diagram/" + dir_name for dir_name in dir_names]
    utils.make_dir(required_dirs)
    for path in csv_paths:
        bi.scatter_diagram_visualize(params.eval_params, path, visualize_dir+"scatter_diagram/")

    # ヒストグラムを作成
    if os.path.isdir(visualize_dir + "histogram/"):
        shutil.rmtree(visualize_dir + "histogram")
    ## csvのファイルパスからdir名を持ってくる
    dir_names = [os.path.splitext(os.path.basename(path))[0] for path in csv_paths]
    ## dir名からdirを生成
    required_dirs = [visualize_dir + "histogram"] + [visualize_dir + "histogram/" + dir_name for dir_name in dir_names]
    utils.make_dir(required_dirs)
    for path in csv_paths:
        bi.histogram_visualize(params.eval_params, path, output_path=visualize_dir+"histogram/")

    # 散布図を結合
    if os.path.isdir(visualize_dir + "concat_scatter_diagram/"):
        shutil.rmtree(visualize_dir + "concat_scatter_diagram")
    dir_name = ''
    for index,path in enumerate(csv_paths):
        dir_name += os.path.splitext(os.path.basename(path))[0]
        if index != len(csv_paths)-1:
            dir_name += '&'
    ## dir名からdirを生成
    required_dirs = [visualize_dir + "concat_scatter_diagram"] + [visualize_dir + "concat_scatter_diagram/" + dir_name]
    utils.make_dir(required_dirs)
    bi.concat_scatter_diagram_visualize(params.eval_params, csv_paths, visualize_dir+"concat_scatter_diagram/"+dir_name+"/")

    # ヒストグラムを結合
    if os.path.isdir(visualize_dir + "concat_histogram/"):
        shutil.rmtree(visualize_dir + "concat_histogram")
    dir_name = ''
    for index,path in enumerate(csv_paths):
        dir_name += os.path.splitext(os.path.basename(path))[0]
        if index != len(csv_paths)-1:
            dir_name += '&'
    ## dir名からdirを生成
    required_dirs = [visualize_dir + "concat_histogram"] + [visualize_dir + "concat_histogram/" + dir_name]
    utils.make_dir(required_dirs)
    bi.concat_histogram_visualize(params.eval_params, csv_paths, visualize_dir+"concat_histogram/"+dir_name+"/")

    # pair plotを作成
    if os.path.isdir(visualize_dir + "pair_plot/"):
        shutil.rmtree(visualize_dir + "pair_plot")
    required_dirs = [visualize_dir + "pair_plot"]
    utils.make_dir(required_dirs)
    bi.pair_plot(params.eval_params, csv_paths, visualize_dir+"pair_plot/")

    logger.info("visualize complete!")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser = common_args(parser)
	args = parser.parse_args()
	params = Parameters(**setup_params(vars(args), args.parameters))  # args，run_date，git_revisionなどを追加した辞書を取得

	# Setting Logger
	result_dir = Path(params.args['eval_graphs']).parent

	# ログ設定
	logger = getLogger(__name__)

	# visualize
	set_logging(str(result_dir), file_name="visualize")  # ログを標準出力とファイルに出力するよう設定
	logger.info("Start : Visualize Graph")
	graph_plot(params)
	logger.info("End : Visualize Graph")
	# graph_visualize(args.eval_graphs+"*.pkl", args.visualize_path)
	# visualize_rep(params=params)
	if params.args["history_vis"] is not None:
		dfs_visualize(params)
	# get_degree_distribution(params)
	# utils.copy_dirs(glob.glob("result/t*/visualize/average_param/"),"result/visualizes/")
	# utils.rename_dirs(sorted(glob.glob("result/*p100s*")))
	# shutil.make_archive('result/zip_vis', 'zip', root_dir="result/visualizes")
