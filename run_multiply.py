"""
evalやvisualizeを条件を変えて複数回実行するためのモジュール.
"""

import os
from logging import getLogger
import argparse
import torch
import shutil
import joblib
import networkx as nx
import pickle
from pathlib import Path
from dataclasses import asdict
import json
import glob

from config import common_args, Parameters
from utils import setup_params, set_logging, fix_seed, make_dir, dump_params
from eval import eval
from visualize import graph_plot
from dimension_reduction import visualize_dimension_reduction
from summarize_results import summarize_visualize

logger = getLogger(__name__)

def eval_multiply(params: Parameters):
    """evalを指定したディレクトリ内で複数回実行する関数.
        Args:
            params (config.Parameters)  : global変数のset
    """
    if params.args['summarize_dir'] is not None:
        set_logging(params.args['summarize_dir'], file_name="eval_multiply")  # ログを標準出力とファイルに出力するよう設定
        eval_models = [
            dirname for dirname in glob.iglob(params.args['summarize_dir'] + '/*/train/valid_best_weight')
        ]
        args_dict = params.args
        for model in sorted(eval_models):
            # logger.info(f"modelは{model}です。")
            result_dir = str(Path(model).parents[1]) # modelの親ディレクトリ
            args_dict.update({'eval_model': model, 'result_dir': result_dir})
            params_in_result_path = result_dir + "/parameters.json"
            params_in_result = Parameters(**setup_params(args_dict, params_in_result_path, update_params={'number_of_generated_samples': params.number_of_generated_samples}))
            if os.path.isfile(os.path.join(result_dir, "parameters_summarize.json")):
                os.remove(os.path.join(result_dir, "parameters_summarize.json"))
            dump_params(params, result_dir, file_name="parameters_summarize.json")
            for s in params.seed_list:
                eval_dir_name = "eval_seed" + str(s)
                new_dir = os.path.join(result_dir, eval_dir_name)
                if os.path.isdir(new_dir):
                    shutil.rmtree(new_dir)
                #     if len(glob.glob(os.path.join(new_dir ,"*.pkl"))) == 3:
                #         logger.info(f"{eval_dir_name}の結果は既に存在します.")
                #         continue
                #     else:
                #         logger.info(f"{eval_dir_name}の結果が不完全です. 再実行します.")
                #         shutil.rmtree(new_dir)
                #         make_dir([new_dir])
                #         # seedの固定
                #         logger.info(f"Seedを{s}に固定します。")
                #         fix_seed(s)
                #         eval(params_in_result, eval_dir_name= eval_dir_name)
                # else:
                make_dir([new_dir])
                # seedの固定
                logger.info(f"Seedを{s}に固定します。")
                fix_seed(s)
                eval(params_in_result, eval_dir_name= eval_dir_name)
    else:
        if params.args['eval_model'] is None:
            logger.info("評価対象のモデルを指定してください.")
            exit()
        result_dir = str(Path(params.args['eval_model']).parents[1])
        params_in_result_path = result_dir + "/parameters.json"
        args_dict = params.args
        args_dict.update({'result_dir': result_dir})
        params_in_result = Parameters(**setup_params(args_dict, params_in_result_path, update_params={'number_of_generated_samples': params.number_of_generated_samples}))
        set_logging(result_dir, file_name="eval_multiple")
        for s in params.seed_list:
            eval_dir_name = "eval_seed" + str(s)
            new_dir = os.path.join(result_dir, eval_dir_name)
            if os.path.isdir(new_dir):
                shutil.rmtree(new_dir)
            #     if len(glob.glob(os.path.join(new_dir ,"*.pkl"))) == 3:
            #         logger.info(f"{eval_dir_name}の結果は既に存在します.")
            #         continue
            #     else:
            #         logger.info(f"{eval_dir_name}の結果が不完全です. 再実行します.")
            #         shutil.rmtree(new_dir)
            #         make_dir([new_dir])
            #         # seedの固定
            #         logger.info(f"Seedを{s}に固定します。")
            #         fix_seed(s)
            #         eval(params_in_result, eval_dir_name= eval_dir_name)
            # else:
            make_dir([new_dir])
            # seedの固定
            logger.info(f"Seedを{s}に固定します。")
            fix_seed(s)
            eval(params_in_result, eval_dir_name=eval_dir_name)

def visualize_multiply(params: Parameters):
    """visualizeを指定したディレクトリ内で複数回実行する関数.
        Args:
            params (config.Parameters)  : global変数のset
    """
    if params.args['summarize_dir'] is not None:
        set_logging(params.args['summarize_dir'], file_name="visualize_multiply")  # ログを標準出力とファイルに出力するよう設定
        logger.info("Start : Visualize Graph")
        logger.info("run multiple")
        eval_dirs = [
            dirname for dirname in glob.iglob(params.args['summarize_dir'] + '/*/eval_seed*')
        ]
        eval_dirs = sorted(eval_dirs)
        args_dict = params.args
        for eval_dir in eval_dirs:
            result_dir = str(Path(eval_dir).parent)
            args_dict.update({'eval_graphs': eval_dir + '/', 'result_dir': result_dir})
            params_in_result_path = result_dir + "/parameters.json"
            params_in_result = Parameters(**setup_params(args_dict, params_in_result_path, update_params={'pickup_num': params.pickup_num}))
            visualize_dir_name = 'visualize_seed' + eval_dir.split('seed')[-1]
            if params.use_unconnected_graphs:
                visualize_dir_name += '_unconnected'
            # if os.path.isdir(os.path.join(result_dir, visualize_dir_name)):
            #     shutil.rmtree(os.path.join(result_dir, visualize_dir_name))
            #     logger.info(f"{visualize_dir_name}の結果は既に存在します.")
            #     continue
            # else:
            make_dir([os.path.join(result_dir, visualize_dir_name)])
            graph_plot(params_in_result, visualize_dir_name)
        
    else:
        if params.args['eval_graphs'] is None:
            logger.info("可視化対象のグラフを指定してください.")
            exit()
        result_dir = str(Path(params.args['eval_graphs']).parent)
        set_logging(result_dir, file_name="visualize_multiple")  # ログを標準出力とファイルに出力するよう設定
        eval_dirs = [
            dirname for dirname in glob.iglob(result_dir + '/eval_seed*')
        ]
        eval_dirs = sorted(eval_dirs)
        args_dict = params.args
        for eval_dir in eval_dirs:
            args_dict.update({'eval_graphs': eval_dir + '/', 'result_dir': result_dir})
            params_in_result_path = result_dir + "/parameters.json"
            params_in_result = Parameters(**setup_params(args_dict, params_in_result_path, update_params={'pickup_num': params.pickup_num}))
            visualize_dir_name = 'visualize_seed' + eval_dir.split('seed')[-1]
            if params.use_unconnected_graphs:
                visualize_dir_name += '_unconnected'
            # if os.path.isdir(os.path.join(result_dir, visualize_dir_name)):
            #     shutil.rmtree(os.path.join(result_dir, visualize_dir_name))
            #     logger.info(f"{visualize_dir_name}の結果は既に存在します.")
            #     continue
            # else:
            make_dir([os.path.join(result_dir, visualize_dir_name)])
            graph_plot(params_in_result, visualize_dir_name)

def dimension_reduction_multiply(params: Parameters):
    """次元削減を指定したディレクトリ内で複数回実行する関数.
        Args:
            params (config.Parameters)  : global変数のset
    """
    if params.args['summarize_dir'] is None:
        logger.info("summarize_dirを指定してください.")
        exit()
    set_logging(params.args['summarize_dir'], file_name="dimension_reduction_multiply")  # ログを標準出力とファイルに出力するよう設定
    logger.info("Start : Dimension Reduction")
    logger.info("run multiple")

    result_dirs_list = [
        dirname for dirname in glob.iglob(params.args['summarize_dir'] + '/*/')
    ]
    result_dirs_list = sorted(result_dirs_list)
    args_dict = params.args
    parameters_list = []
    for result_dir in result_dirs_list:
        args_dict.update({'result_dir': result_dir})
        parameters_list.append(Parameters(**setup_params(args_dict, "parameters.json")))

    for params_in_result in parameters_list:
        visualize_dimension_reduction(params_in_result, logger)

def summarize_multiply(params: Parameters):
    """summarize_resultsを指定したディレクトリ内で複数回実行する関数.
        Args:
            params (config.Parameters)  : global変数のset
    """
    if params.args['summarize_dir'] is None:
        logger.info("summarize_dirを指定してください.")
        exit()
    set_logging(params.args['summarize_dir'], file_name="summarize_multiply")  # ログを標準出力とファイルに出力するよう設定
    summarize_dirs_list = [
        dirname for dirname in glob.iglob(params.args['summarize_dir'] + '/**/**/') if '2rw' in dirname and 'csv' not in dirname
    ]

    logger.info("Start : Summarize Results")
    logger.info("run multiple")
    heatmap_dir_path = params.args['heatmap_dir']
    # if os.path.isdir(heatmap_dir_path):
    #     shutil.rmtree(heatmap_dir_path)
    make_dir([heatmap_dir_path, heatmap_dir_path + '/rmse', heatmap_dir_path + '/unconnected_ratio'])

    for summarize_dir in summarize_dirs_list:
        args_dict = params.args
        args_dict.update({'summarize_dir': summarize_dir, 'heatmap_dir': heatmap_dir_path})
        result_dirs_list = [
            dirname for dirname in glob.iglob(summarize_dir + '/*/') if 'csv' not in dirname
        ]
        params_update = Parameters(**setup_params(args_dict, sorted(result_dirs_list)[0] + "/parameters.json"))
        summarize_visualize(params_update, mode='2nd_rw')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = common_args(parser)
    args = parser.parse_args()
    params = Parameters(**setup_params(vars(args), args.parameters))  # args，run_date，git_revisionなどを追加した辞書を取得

    # ログ設定
    logger = getLogger(__name__)

    if 'eval' in params.args['multiply_mode']:
        eval_multiply(params)
    if 'visualize' in params.args['multiply_mode']:
        visualize_multiply(params)
    if 'dimension_reduction' in params.args['multiply_mode']:
        dimension_reduction_multiply(params)
    if 'summarize' in params.args['multiply_mode']:
        summarize_multiply(params)
