import argparse
import json
import logging
import os
import sys
from typing import List, Tuple

import numpy as np
import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import Parameters, common_args
from preprocess import preprocess
from train import train_cvae
from utils import setup_params, make_dir, dump_params, fix_seed, set_logging


@pytest.fixture
def param_json(tmpdir) -> str:
    """trainに必要なパラメータをtmpdirにjsonファイルとして作成し、そのファイル名を返す"""
    params = {
        "gpu_id": 0,
        "conditional_mode": True,
        "condition_params": [
            "Average path length"
        ],
        "condition_round": 4,
        "condition_size": 1,
        "condition_values": {
            "Power-law exponent": [
                2.6,
                3.0,
                3.4
            ],
            "Clustering coefficient": [
                0.1,
                0.2,
                0.3
            ],
            "Average path length": [
                3.0,
                4.0,
                5.0
            ],
            "Average degree": [
                3,
                4,
                5
            ],
            "Edge density": [
                0.05,
                0.075,
                0.1
            ],
            "Modularity": [
                0.5,
                0.6,
                0.7
            ],
            "Diameter": [
                10,
                20,
                30
            ],
            "Largest component size": [
                7.0,
                14.0,
                20.0
            ]
        },
        "dataset_size": 2000,
        "number_of_nodes": 50,
        "connection_new_edge": 3,
        "connection_neighbor_node": 3,
        "rewiring_edge_probs": 0.3,
        "edge_create_prob": 0.5,
        "split_size": {
            "train": 0.9,
            "valid": 0.1,
            "test": 0
        },
        "dfs_mode": "high_degree_first",
        "ignore_label": 1500,
        "normalize": False,
        "standardize": False,
        "reddit_path": "./data/reddit_threads/reddit_edges.json",
        "twitter_path": "./data/edgelists_50/renum*",
        "base_dataset_path": "./dataset/",
        "twitter_augment_dataset_path": "augment_dataset/twitter_augment_30%_edge50_apl4_gene100_lm300.joblib",
        "apl_uniform_dataset_path": "./dataset_by_generater/graphs_for_fe_train.joblib",
        "generated_and_twitter_path": "./dataset_by_generater/relabeled_GraphTune_generated(3_4_5)+twitter.joblib",
        "augment_mode": "edge_random",
        "edge_random_p": 0.3,
        "gene_graph_num": 100,
        "loop_max": 300,
        "apl_th": 4.0,
        "search": "dfs",
        "power_degree_border_line": 0.7,
        "fe_preprocess_onehot": True,
        "fe_preprocess_convert_graph_obj": True,
        "model_set": [
            "cvae",
            "cvae_with_fe",
            "fe",
            "cvae_for_2_tuples"
        ],
        "froze_fe_params": True,
        "dropout": 0.5,
        "model_params": {
            "batch_size": 37,
            "clip_th": 1.0,
            "lr": 0.001,
            "weight_decay": 0,
            "emb_size": 227,
            "en_hidden_size": 223,
            "rep_size": 10,
            "de_hidden_size": 250,
            "word_drop": 0,
            "fe_emb_size": 256,
            "fe_hidden_size": 512,
            "fe_rep_size": 1,
            "fe_lstm_num_layer": 1,
            "fe_dropout": 0.294456,
            "alpha": 1,
            "beta": 3,
            "gamma": 300,
            "encoder_condition": True,
            "decoder_h_c_condition": True,
            "decoder_sequence_condition": True
        },
        "epochs": 3,
        "model_save_point": 10,
        "early_stopping_rounds": 10,
        "eval_params": [
            "Power-law exponent",
            "Clustering coefficient",
            "Average path length",
            "Average degree",
            "Edge density",
            "Modularity",
            "Diameter",
            "Largest component size"
        ],
        "sampling_generation": True,
        "softmax_temperature": 1.0,
        "temperature_list": [
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            1.7,
            1.8,
            1.9,
            2.0,
            2.1,
            2.2,
            2.3,
            2.4,
            2.5,
            2.6,
            2.7,
            2.8,
            2.9,
            3.0,
            3.1,
            3.2,
            3.3,
            3.4,
            3.5,
            3.6,
            3.7,
            3.8,
            3.9,
            4.0
        ],
        "logging_condition_by_graphtune_fe": True,
        "logging_graph_vis_with_dataset_and_decoding": True,
        "acc_range": {
            "Power-law exponent": [
                0.1,
                0.1
            ],
            "Clustering coefficient": [
                0.01,
                0.01
            ],
            "Average path length": [
                0.05,
                0.05
            ],
            "Average degree": [
                0.1,
                0.1
            ],
            "Edge density": [
                0.005,
                0.005
            ],
            "Modularity": [
                0.02,
                0.02
            ],
            "Diameter": [
                0,
                0
            ],
            "Largest component size": [
                0,
                0
            ]
        },
        "seed": 200,
    }

    file_name = "test_parameters.json"
    with open(tmpdir + file_name, "w") as f:
        json.dump(params, f, ensure_ascii=False, indent=4)

    return file_name


@pytest.fixture
def train_cvae_args(tmpdir, param_json) -> Tuple['config.Parameters', 'logging.Logger', float]:
    """train関数を呼び出す前に行う処理
    """
    # args, paramsの設定
    args = {
            "parameters": "parameters.json",
            "dataset": "twitter",
            "preprocess": True,
            "preprocess_type": "dfs_5_tuples",
            "softmax": True,
            "run_multiple": False,
            "checkpoint_file": None, 
            "init_epoch": None,
            "use_model": "cvae"
    }
    params = Parameters(**setup_params(args, tmpdir + param_json))  # args，run_date，git_revisionなどを追加した辞書を取得

    # 結果出力用のdirを作成
    result_dir = params.result_dir
    required_dirs = [result_dir, result_dir + "/train", result_dir + "/train/csv", result_dir + "/eval",
                    result_dir + "/test", result_dir + "/visualize", result_dir + "/visualize/csv",
                    result_dir+"/dataset"]
    make_dir(required_dirs)

    # パラメーターをdump
    dump_params(params, f'{result_dir}')

    # loggingの設定
    logger = logging.getLogger(__name__)
    set_logging(result_dir, file_name="train")  # ログを標準出力とファイルに出力するよう設定

    # 期待する結果データを作成
    expected = 273378.98046875

    yield params, logger, expected

    tmpdir.remove()


def test_train_cvae(train_cvae_args):
    assert int(train_cvae(params=train_cvae_args[0], logger=train_cvae_args[1])) == int(train_cvae_args[2])
