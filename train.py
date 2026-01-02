"""
Neural networkを使用したmodelを学習するためのモジュール.
Supports LSTM, Transformer, and Hybrid architectures.

(1) GraphTuneの学習
    train_cvae()
"""
# Python標準モジュール
import time
import datetime
import os
import argparse
from logging import getLogger, Logger
import shutil
import joblib
import math
# Third-partyモジュール
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchinfo import summary
# 自作モジュール
from config import common_args, Parameters
import utils
from utils import dump_params, setup_params, seed_worker
from utils import set_logging, make_dir
from preprocess import preprocess
from models import cvae, cvae_for_2_tuples
from models.transformer_cvae import TransformerCVAE, create_padding_mask_from_labels  # UPDATED: helper
from models.hybrid_cvae import HybridCVAE  # Hybrid CVAE
from graph_process import graph_utils, graph_statistic


def train_cvae(params: Parameters, logger: Logger) -> float:
    """CVAEを学習する関数

        Args:
            params (config.Parameters)  : global変数のset
            logger (logging.Logger)	    : logging
        
        Returns:
            (float): valid minimum loss
    """
    # Seed値の固定
    utils.fix_seed(params.seed)

    # Tensorboardの設定
    writer = SummaryWriter(log_dir=f"./{params.result_dir}")

    # Open epoch毎lossが記載されるcsv file
    with open(f"./{params.result_dir}/train/csv/train_loss_data.csv", "w") as csv_train_loss:
        csv_train_loss.write(f"epoch,loss\n")
    with open(f"./{params.result_dir}/train/csv/valid_loss_data.csv", "w") as csv_valid_loss:
        csv_valid_loss.write(f"epoch,loss\n")

    # deviceの設定
    device = torch.device(f"cuda:{params.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(params.gpu_id)

    # 結果出力用dir
    result_dir = f'{params.result_dir}/'

    # 前処理
    if params.args['preprocess']:
        logger.info("start preprocess...")

        preprocess_dirs = [
            result_dir + "dataset/train",
            result_dir + "dataset/valid",
            result_dir + "dataset/test"
        ]
        make_dir(preprocess_dirs)
        preprocess(
            params=params,
            train_directory=result_dir + "dataset/train/",
            valid_directory=result_dir + "dataset/valid/",
            test_directory=result_dir + "dataset/test/"
        )

    # 前処理後のdataをload
    train_dataset       = joblib.load(result_dir + "dataset/train/onehot")
    train_label         = joblib.load(result_dir + "dataset/train/label")
    train_conditional   = joblib.load(result_dir + "dataset/train/conditional")
    valid_dataset       = joblib.load(result_dir + "dataset/valid/onehot")
    valid_label         = joblib.load(result_dir + "dataset/valid/label")
    valid_conditional   = joblib.load(result_dir + "dataset/valid/conditional")

    # ★★ FIXED HERE: paramが4個でも5個でも動くようにした ★★
    param_obj = joblib.load(result_dir + "dataset/param")
    if isinstance(param_obj, (list, tuple)):
        if len(param_obj) == 4:
            time_size, node_size, edge_size, conditional_size = param_obj
            max_seq_len = None
        else:
            # 新しい前処理: (time_size, node_size, edge_size, conditional_size, max_sequence_length, ...)
            time_size, node_size, edge_size, conditional_size = param_obj[:4]
            max_seq_len = param_obj[4]
            logger.info(f"max_sequence_length = {max_seq_len}")
    else:
        # 念のため: もしdictや他の形式で来た場合
        raise ValueError("dataset/param の形式が想定外です。list/tuple を想定しています。")

    # labelをdeviceに乗せる
    train_label = [element.to(device) for element in train_label]
    valid_label = [element.to(device) for element in valid_label]

    # datasetを作成
    use_model = params.args['use_model']

    # DFSの1 stepあたりのサイズをそろえる処理（LSTMでもTransformerでも同じ）
    if use_model in ["cvae", "transformer_cvae", "hybrid_cvae"]:
        dfs_size = 2 * time_size + 2 * node_size + edge_size + conditional_size
        dfs_size_list = [time_size, time_size, node_size, node_size, edge_size]

        # 条件を系列長にあわせて繰り返して concat
        train_conditional = torch.cat(
            [train_conditional for _ in range(train_dataset.shape[1])],
            dim=1
        ).unsqueeze(2)
        valid_conditional = torch.cat(
            [valid_conditional for _ in range(valid_dataset.shape[1])],
            dim=1
        ).unsqueeze(2)

        train_dataset = torch.cat((train_dataset, train_conditional), dim=2)
        valid_dataset = torch.cat((valid_dataset, valid_conditional), dim=2)

        if use_model == "cvae":
            logger.info(
                f"\n--------------\ntime size: {time_size}\nnode size: {node_size}\nedge size: {edge_size}\nconditional size: {conditional_size}\n--------------\n"
            )
        elif use_model == "transformer_cvae":
            logger.info(
                f"\n--------------\nTransformer CVAE Model\ntime size: {time_size}\nnode size: {node_size}\nedge size: {edge_size}\nconditional size: {conditional_size}\n--------------\n"
            )
        elif use_model == "hybrid_cvae":
            logger.info(
                f"\n--------------\nHybrid CVAE Model (LSTM Encoder + Transformer Decoder)\ntime size: {time_size}\nnode size: {node_size}\nedge size: {edge_size}\nconditional size: {conditional_size}\n--------------\n"
            )

    elif use_model == "cvae_for_2_tuples":
        dfs_size = 2 * time_size + conditional_size
        train_conditional = torch.cat(
            [train_conditional for _ in range(train_dataset.shape[1])],
            dim=1
        ).unsqueeze(2)
        valid_conditional = torch.cat(
            [valid_conditional for _ in range(valid_dataset.shape[1])],
            dim=1
        ).unsqueeze(2)
        train_dataset = torch.cat((train_dataset, train_conditional), dim=2)
        valid_dataset = torch.cat((valid_dataset, valid_conditional), dim=2)
        logger.info(
            f"\n--------------\ntime size: {time_size}\nnode size: {node_size}\nedge size: {edge_size}\nconditional size: {conditional_size}\n--------------\n"
        )
    else:
        logger.info("そのようなモデルは存在しません.")
        exit()

    # Dataloaderのseed固定
    g = torch.Generator()
    g.manual_seed(params.seed)

    # dataloaderを作成
    data_num = {"train": train_dataset.shape[0], "valid": valid_dataset.shape[0]}
    logger.info(f"Dataset size: {data_num}")
    train_label_args = torch.LongTensor(list(range(data_num["train"])))
    valid_label_args = torch.LongTensor(list(range(data_num["valid"])))
    train_dl = DataLoader(
        TensorDataset(train_label_args, train_dataset),
        shuffle=True,
        batch_size=params.model_params["batch_size"],
        drop_last=False,
        pin_memory=True,
        generator=g,
        worker_init_fn=seed_worker
    )
    valid_dl = DataLoader(
        TensorDataset(valid_label_args, valid_dataset),
        shuffle=False,
        batch_size=params.model_params["batch_size"],
        pin_memory=True,
        generator=g,
        worker_init_fn=seed_worker
    )

    # 学習するモデルの定義
    if use_model == "cvae":
        model = cvae.CVAE(dfs_size, time_size, node_size, edge_size, conditional_size, params, device)
        logger.info("Using LSTM-based CVAE model")

    elif use_model == "transformer_cvae":
        model = TransformerCVAE(dfs_size, time_size, node_size, edge_size, conditional_size, params, device)
        logger.info("Using Transformer-based CVAE model (with padding mask support)")

    elif use_model == "hybrid_cvae":
        model = HybridCVAE(dfs_size, time_size, node_size, edge_size, conditional_size, params, device)
        logger.info("Using Hybrid CVAE model (LSTM Encoder + Transformer Decoder)")

    elif use_model == "cvae_for_2_tuples":
        model = cvae_for_2_tuples.CVAE(
            dfs_size=dfs_size,
            time_size=time_size,
            condition_size=conditional_size,
            params=params,
            device=device
        )
        logger.info("Using LSTM-based CVAE for 2-tuples model")

    # (Optional) チェックポイントをload
    if params.args['checkpoint_file'] and params.args['init_epoch']:
        model.load_state_dict(torch.load(params.args['checkpoint_file'], map_location="cpu"))
        init_epoch = int(params.args['init_epoch'])
        logger.info(f"Loaded checkpoint from {params.args['checkpoint_file']}, starting from epoch {init_epoch}")
    else:
        init_epoch = 1

    # modelをGPUに乗せる
    model = model.to(device)

    # モデル概要をlogging
    # Transformer/Hybridは src_key_padding_mask が必要なので summary() はやめておく
    if use_model in ["cvae", "cvae_for_2_tuples"]:
        logger.info("モデル概要")
        logger.info(
            summary(
                model,
                input_size=(params.model_params["batch_size"], train_dataset.shape[1], train_dataset.shape[2]),
                col_names=["output_size", "num_params"],
                device=device
            )
        )
    else:
        # Transformer系は構造だけログに出す
        logger.info("モデル概要 (printed):")
        logger.info(model)

    # 最適関数の定義
    opt = optim.Adam(model.parameters(), lr=params.model_params["lr"])
    logger.info("最適化関数")
    logger.info(opt)

    # NEW: Create padding masks ONLY for transformer/hybrid
    padding_mask = {"train": None, "valid": None}
    if use_model in ["transformer_cvae", "hybrid_cvae"]:
        # ここでは label の 0 番目 (t_u) を使ってマスクを作る
        train_padding_mask = create_padding_mask_from_labels(train_label[0], params.ignore_label)
        valid_padding_mask = create_padding_mask_from_labels(valid_label[0], params.ignore_label)
        # ここではCPUのまま持っておいて、各batchで .to(device) する
        padding_mask = {"train": train_padding_mask, "valid": valid_padding_mask}
        logger.info("Created padding masks for Transformer/Hybrid models")

    # 学習開始
    logger.info("Start training ...")
    start_time = datetime.datetime.now().replace(microsecond=0)
    with open(f'{params.result_dir}/train_time.csv', 'w') as f:
        f.write('model name, processing time, model path, time stamp \n')

    train_min_loss = 1e10
    valid_min_loss = 1e10
    train_best_epoch = init_epoch
    valid_best_epoch = init_epoch

    dataloader = {"train": train_dl, "valid": valid_dl}
    label      = {"train": train_label, "valid": valid_label}
    condition  = {"train": train_conditional, "valid": valid_conditional}

    for epoch in range(init_epoch, 1 + params.epochs, 1):
        logger.info("Epoch: [%d/%d]:" % (epoch, params.epochs))

        # 各epochにlossを0で初期化する
        encoder_loss_per_epoch      = {"train": 0., "valid": 0.}
        five_tuples_dict            = {"tu": 0., "tv": 0., "lu": 0., "lv": 0., "le": 0.}
        decoder_loss_per_epoch_dict = {"train": five_tuples_dict.copy(), "valid": five_tuples_dict.copy()}
        decoder_acc_per_epoch_dict  = {"train": five_tuples_dict.copy(), "valid": five_tuples_dict.copy()}
        model_loss_per_epoch        = {"train": 0., "valid": 0.}

        # 訓練phaseと検証phase
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
                logger.info("  train:")
            else:
                model.eval()
                logger.info("  valid:")

            # データをイテレート
            for i, (indicies, data) in enumerate(dataloader[phase], 0):
                opt.zero_grad()
                data = data.to(device)

                # バッチ用マスクを用意（Transformer/Hybridのみ）
                if use_model in ["transformer_cvae", "hybrid_cvae"]:
                    batch_mask = padding_mask[phase][indicies].to(device)
                else:
                    batch_mask = None  # LSTM系は使わない

                with torch.set_grad_enabled(phase == 'train'):
                    # Forward
                    if use_model in ["transformer_cvae", "hybrid_cvae"]:
                        mu, sigma, tu, tv, lu, lv, le = model(data, src_key_padding_mask=batch_mask)
                    elif use_model == "cvae":
                        mu, sigma, tu, tv, lu, lv, le = model(data)
                    elif use_model == "cvae_for_2_tuples":
                        mu, sigma, tu, tv = model(data)

                    # Encoder loss
                    encoder_loss = model.encoder.loss(mu, sigma)
                    encoder_loss_per_epoch[phase] += encoder_loss.item()

                    # Decoder loss
                    if use_model in ["cvae", "transformer_cvae", "hybrid_cvae"]:
                        results = {"tu": tu, "tv": tv, "lu": lu, "lv": lv, "le": le}
                        targets = {
                            "tu": label[phase][0][indicies],
                            "tv": label[phase][1][indicies],
                            "lu": label[phase][2][indicies],
                            "lv": label[phase][3][indicies],
                            "le": label[phase][4][indicies],
                        }
                        decoder_loss_dict, decoder_total_loss = model.decoder.loss(results, targets)
                        for (key, val) in decoder_loss_dict.items():
                            decoder_loss_per_epoch_dict[phase][key] += val.item()
                    elif use_model == "cvae_for_2_tuples":
                        results = {"tu": tu, "tv": tv}
                        targets = {
                            "tu": label[phase][0][indicies],
                            "tv": label[phase][1][indicies],
                        }
                        decoder_loss_dict, decoder_total_loss = model.decoder.loss(results, targets)
                        for (key, val) in decoder_loss_dict.items():
                            decoder_loss_per_epoch_dict[phase][key] += val.item()

                    # Model loss
                    model_loss = model.loss(encoder_loss, decoder_total_loss)
                    model_loss_per_epoch[phase] += model_loss.item()

                    # Accuracy
                    acc_dict = model.decoder.accuracy(results, targets)
                    for (key, score) in acc_dict.items():
                        decoder_acc_per_epoch_dict[phase][key] += score

                    # nan check
                    if torch.isnan(model_loss):
                        logger.info(f"model lossにnanを検知.")
                        torch.save(model.state_dict(),
                                   params.result_dir + "/train/nan_iter_" + str(i) + "_weight_" + str(epoch))
                        torch.save(opt.state_dict(),
                                   params.result_dir + "/train/nan_opt_iter_" + str(i) + "_weight_" + str(epoch))
                        writer.close()
                        logger.info(f"train best epoch : {train_best_epoch}")
                        logger.info(f"valid best epoch : {valid_best_epoch}")
                        exit()

                    # Backward
                    if phase == "train":
                        model_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), params.model_params["clip_th"])
                        opt.step()

            # Save loss / acc
            writer.add_scalar(f"{phase}_loss/encoder_loss",
                              encoder_loss_per_epoch[phase] / data_num[phase], epoch)
            logger.info(f"    encoder_loss = {encoder_loss_per_epoch[phase] / data_num[phase]}")

            for (key, val) in decoder_loss_per_epoch_dict[phase].items():
                writer.add_scalar(f"{phase}_loss/{key}_loss", val / data_num[phase], epoch)
                logger.info(f"    {key}_loss = {val / data_num[phase]}")

            for (key, val) in decoder_acc_per_epoch_dict[phase].items():
                writer.add_scalar(
                    f"{phase}_acc/{key}_acc",
                    val / math.ceil(data_num[phase] / params.model_params['batch_size']),
                    epoch
                )
                logger.info(
                    f"    {key}_acc = {val / math.ceil(data_num[phase] / params.model_params['batch_size'])}"
                )

            writer.add_scalar(f"{phase}_loss/model_loss",
                              model_loss_per_epoch[phase] / data_num[phase], epoch)
            logger.info(f"    model_loss = {model_loss_per_epoch[phase] / data_num[phase]}")
            with open(f"./{params.result_dir}/train/csv/{phase}_loss_data.csv", "a") as csv_loss:
                csv_loss.write(f"{epoch},{model_loss_per_epoch[phase] / data_num[phase]}\n")

        # Checkpoint
        if epoch % params.model_save_point == 0:
            checkpoint_time = datetime.datetime.now().replace(microsecond=0)
            processing_time = checkpoint_time - start_time
            with open(f'{params.result_dir}/train_time.csv', 'a') as f:
                f.write(f'{use_model}_{epoch}, {processing_time}, ' +
                        params.result_dir + "/train/weight_" + str(epoch) +
                        ", " + str(checkpoint_time) + '\n')
            torch.save(model.state_dict(), params.result_dir + "/train/weight_" + str(epoch))
            logger.info(f'Checkpoint: {epoch}')

        # Save train best
        if model_loss_per_epoch["train"] < train_min_loss:
            train_best_epoch = epoch
            train_min_loss = model_loss_per_epoch["train"]
            torch.save(model.state_dict(), params.result_dir + "/train/train_best_weight")
            logger.info(f'  Update train best epoch: {epoch}')
        logger.info(f"train best epoch : {train_best_epoch}")

        # Save valid best
        if model_loss_per_epoch["valid"] < valid_min_loss:
            valid_best_epoch = epoch
            valid_min_loss = model_loss_per_epoch["valid"]
            torch.save(model.state_dict(), params.result_dir + "/train/valid_best_weight")
            logger.info(f'  Update valid best epoch: {epoch}')
        logger.info(f"valid best epoch : {valid_best_epoch}")

        # Save last model
        if params.save_last_model:
            last_path_prev = f"result/{params.run_date}/train/last_weight_{epoch-1}"
            if os.path.isfile(last_path_prev):
                os.remove(last_path_prev)
            last_path = f"result/{params.run_date}/train/last_weight_{epoch}"
            torch.save(model.state_dict(), last_path)
            logger.info(f"Save last model")

    writer.close()
    logger.info(f"train best epoch : {train_best_epoch}")
    logger.info(f"valid best epoch : {valid_best_epoch}")
    logger.info("train complete!")
    
    return valid_min_loss


if __name__ == "__main__":
    # 引数やGlobal変数を設定
    parser = argparse.ArgumentParser()
    parser = common_args(parser)
    args = parser.parse_args()
    params = Parameters(**setup_params(vars(args), args.parameters))

    # 結果出力用ディレクトリの作成
    result_dir = params.result_dir
    required_dirs = [
        result_dir,
        result_dir + "/train",
        result_dir + "/train/csv",
        result_dir + "/eval",
        result_dir + "/test",
        result_dir + "/visualize",
        result_dir + "/visualize/csv",
        result_dir + "/dataset"
    ]
    if params.args['init_epoch'] is None:
        make_dir(required_dirs)
        dump_params(params, f'{result_dir}')
    
    # ログ設定
    logger = getLogger(__name__)
    set_logging(result_dir, file_name="train")

    # グローバル変数をlog出力
    logger.info('parameters: ')
    logger.info(params)

    # train
    print(train_cvae(params=params, logger=logger))

