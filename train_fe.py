# Python標準モジュール
import os
import argparse
import logging
import shutil
import joblib
import math
import copy
from typing import Any
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
from torchviz import make_dot
# 自作モジュール
from config import common_args, Parameters
from models import fe
from preprocess import preprocess
from utils import dump_params, setup_params
from utils import set_logging, make_dir, fix_seed



def setup_logging(params:'config.Parameters') -> Any:
    """tensorboardのSummaryWriterやcsv出力の初期設定を行う関数.

    Returns:
        Any: SummaryWriterインスタンス
    """
    # Tensorboardの設定
    writer = SummaryWriter(log_dir=params.result_dir)

    for phase in ["train", "valid"]:
        # epoch毎にlossが記載されるcsv
        with open(f"{params.result_dir}/train/csv/{phase}_loss_data.csv", "w") as f:
            f.write(f"epoch,loss\n")
        # epoch毎のRMSEが記載されるcsv
        with open(f"{params.result_dir}/train/csv/{phase}_rmse_data.csv", "w") as f:
            f.write(f"epoch,loss\n")
        # modelが推論した値と正解ラベルが記載されたcsv
        with open(f"{params.result_dir}/train/csv/{phase}_pred_data.csv", "w") as f:
            f.write(f"epoch,index,pred,correct\n")

    return writer


def detect_nan(params, loss, model, epoch, prev_model=None, prev_opt=None) -> bool:
    """torch.Tensor型のデータから、nanになっているデータがあるかどうかを検出する

    Args:
        params (config.Parameters): グローバル変数のセット
        loss        (torch.Tensor): loss
        model       (torch.Tensor): model
        epoch                (int): epoch
        prev_model  (torch.Tensor): 1[epoch]前のモデル
        prev_opt    (torch.Tesnor): 1[epoch]前のoptimizer

    Returns:
        (bool): nanを検出したらTrue, それ以外はFalse
    """
    if torch.isnan(loss):
        # nanを検知した時点でのmodelとoptを保存する
        torch.save(model.state_dict(),
            params.result_dir + "/train/nan_iter_" + str(i) + "_weight_" + str(epoch))
        torch.save(opt.state_dict(),
            params.result_dir + "/train/nan_opt_iter_" + str(i) + "_weight_" + str(epoch))
        return True
    else:
        return False


def train_fe(params:'config.Parameters', logger:'logging.Logger', logging_result:bool=True) -> float:
    """FEを学習する関数

        Args:
            params (config.Parameters)  : global変数のset
            logger (logging.Logger)	    : logging
            (Optional)logging_result (bool) :   tensorboardによるSummaryWriterやcsvへのlossの出力をするかどうかのフラグ.
                                                デフォルトはTrue.

        Returns:
            (float) : validation minimum loss [/graph]
    """
    # Seed値の固定
    fix_seed(params.seed)

    # deviceの設定
    device = torch.device(f"cuda:{params.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(params.gpu_id)

    # 学習結果のloggingなど
    if logging_result:
        writer = setup_logging(params)

    # 結果出力用dir
    result_dir = params.result_dir

    # 前処理
    if params.args['preprocess']:
        logger.info("start preprocess...")
        preprocess_dirs = [ result_dir + "dataset/train",
                            result_dir + "dataset/valid",
                            result_dir + "dataset/test"
                        ]
        make_dir(preprocess_dirs)
        preprocess( params=params,
                    train_directory=result_dir + "dataset/train/",
                    valid_directory=result_dir + "dataset/valid/",
                    test_directory =result_dir + "dataset/test/"
                )

    # 前処理後のdataをload
    train_dataset       = joblib.load(result_dir + "dataset/train/onehot")
    train_label         = joblib.load(result_dir + "dataset/train/label")
    train_conditional   = joblib.load(result_dir + "dataset/train/conditional")
    valid_dataset       = joblib.load(result_dir + "dataset/valid/onehot")
    valid_label         = joblib.load(result_dir + "dataset/valid/label")
    valid_conditional   = joblib.load(result_dir + "dataset/valid/conditional")
    time_size, node_size, edge_size, conditional_size = joblib.load(result_dir + "dataset/param")

    # labelをdeviceに乗せる
    train_label = [element.to(device) for element in train_label]
    valid_label = [element.to(device) for element in valid_label]

    # datasetを作成
    dfs_size = 2 * time_size + 2 * node_size + edge_size + conditional_size
    train_conditional = torch.cat([train_conditional for _ in range(train_dataset.shape[1])], dim=1).unsqueeze(2)
    valid_conditional = torch.cat([valid_conditional for _ in range(valid_dataset.shape[1])], dim=1).unsqueeze(2)
    logger.info(f"\n--------------\ntime size: {time_size}\nnode size: {node_size}\nedge size: {edge_size}\nconditional size: {conditional_size}\n--------------\n")

    # Dataloaderのseed固定
    g = torch.Generator()
    g.manual_seed(params.seed)

    # dataloaderを作成
    data_num = {"train": train_dataset.shape[0], "valid": valid_dataset.shape[0]}
    train_label_args = torch.LongTensor(list(range(data_num["train"])))
    valid_label_args = torch.LongTensor(list(range(data_num["valid"])))
    train_dl = DataLoader(
        TensorDataset(train_label_args, train_dataset),
        shuffle=True,										# epoch毎にdataがshuffleされる
        batch_size=params.model_params["batch_size"],		# mini-batchサイズ
        drop_last=False,									# 指定したbacth_sizeでdataを割り切れなかった時、最後のバッチをdropしない
        pin_memory=True,									# TensorをCUDAのpinされたメモリへコピーする
        generator=g											# 乱数生成器を指定
    )
    valid_dl = DataLoader(
        TensorDataset(valid_label_args, valid_dataset),
        shuffle=False,
        batch_size=params.model_params["batch_size"],
        pin_memory=True,
        generator=g
    )

    # 学習するモデルの定義
    model = fe.FeatureEstimator(dfs_size - conditional_size, params, device)
    model = model.to(device)  # modelをGPUに乗せる

    # (Optional) チェックポイントをload
    if params.args['checkpoint_file'] and params.args['init_epoch']:
        model.load_state_dict(torch.load(params.args['checkpoint_file'], map_location="cpu"))
        init_epoch = int(params.args['init_epoch'])  # 初期エポック数
    else:
        init_epoch = 1

    # modelの概要をlogging
    logger.info("モデル概要")
    logger.info(
        summary(
            model,
            input_size=(params.model_params["batch_size"], train_dataset.shape[1], train_dataset.shape[2]),
            col_names=["output_size", "num_params"],
            device=device
        )
    )

    # 最適関数の定義
    opt = optim.Adam(model.parameters(), lr=params.model_params["lr"])
    logger.info("最適化関数")
    logger.info(opt)


    # 学習開始
    train_min_loss      = 1e10  # trainでのmodel_lossの最小値
    valid_min_loss      = 1e10  # validでのmodel_lossの最小値
    train_best_epoch    = init_epoch  # trainでの最もmodel_lossが小さい時のepoch
    valid_best_epoch    = init_epoch  # validでの最もmodel_lossが小さい時のepoch
    early_stopping_cnt  = 0  # epochカウント. valid lossが更新されたら, 0にリセットされる

    dataloader  = {"train": train_dl, "valid": valid_dl}  # DataLoaderのdict
    label       = {"train": train_label, "valid": valid_label}  # ラベルのdict
    condition   = {"train": train_conditional, "valid": valid_conditional}    # Conditionのdict

    logger.info("Start training ...")
    for epoch in range(init_epoch, params.epochs + 1, 1):
        logger.info("Epoch: [%d/%d]:" % (epoch, params.epochs))

        # 各epochにlossを0で初期化する
        model_loss_per_epoch = {"train": 0., "valid": 0.}

        # 訓練phaseと検証phase
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()  # modelを訓練するモードに設定する
                logger.info("  train:")
            else:
                model.eval()  # modelを評価するモードに設定する
                logger.info("  valid:")

            # データをイレテー卜する
            for i, (indicies, data) in enumerate(dataloader[phase], 0):
                opt.zero_grad()  # パラメータの勾配をゼロにします
                data = data.to(device)

                # 訓練の時だけ、履歴を保持する
                with torch.set_grad_enabled(phase == 'train'):

                    # Forward propagate
                    graph_feature = model(data)

                    # Model loss
                    ## model_lossは, 各lossのミニバッチのsumに, 対応するalphaなどの定数を乗じて, 和をとったものである.
                    graph_feature = torch.squeeze(graph_feature, 1)
                    target_fe = condition[phase][indicies].to(device)
                    target_fe = target_fe.transpose(1, 0)[0]
                    model_loss = model.loss(graph_feature, target_fe)
                    model_loss_per_epoch[phase] += model_loss.item()

                    # 訓練の時だけ, 誤差逆伝搬 + 勾配クリッピング + オプティマイズする
                    if phase == "train":
                        model_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), params.model_params["clip_th"])
                        opt.step()

            # Save loss, output
            logger.info(f"    model_loss = {model_loss_per_epoch[phase] / data_num[phase]} [hop^2]")
            logger.info(f"    model_rmse_loss = {math.sqrt(model_loss_per_epoch[phase] / data_num[phase])} [hop]")
            if logging_result:
                writer.add_scalar(f"{phase}_loss/model_loss", model_loss_per_epoch[phase] / data_num[phase], epoch)
                writer.add_scalar(f"{phase}_loss/model_RMSE_loss", math.sqrt(model_loss_per_epoch[phase] / data_num[phase]), epoch)
                with open(f"{params.result_dir}/train/csv/{phase}_loss_data.csv", "a") as f:
                    f.write(f"{epoch},{model_loss_per_epoch[phase] / data_num[phase]}\n")
                with open(f"{params.result_dir}/train/csv/{phase}_rmse_data.csv", "w") as f:
                    f.write(f"{epoch},{math.sqrt(model_loss_per_epoch[phase] / data_num[phase])}\n")


        # Save model at checkpoint
        if epoch % params.model_save_point == 0:
            torch.save(model.state_dict(), params.result_dir + "/train/weight_" + str(epoch))
            logger.info(f'Checkpoint: {epoch}')

        # Save train best model
        if model_loss_per_epoch["train"] < train_min_loss:
            train_best_epoch = epoch
            train_min_loss = model_loss_per_epoch["train"]
            torch.save(model.state_dict(), params.result_dir + "/train/train_best_weight")
            logger.info(f'  Update train best epoch: {epoch}')

        # Save valid best model
        if model_loss_per_epoch["valid"] < valid_min_loss:
            early_stopping_cnt = 0
            valid_best_epoch = epoch
            valid_min_loss = model_loss_per_epoch["valid"]
            torch.save(model.state_dict(), params.result_dir + "/train/valid_best_weight")
            logger.info(f'  Update valid best epoch: {epoch}')
        else:
            early_stopping_cnt += 1

        # early stopping
        if early_stopping_cnt >= params.early_stopping_rounds:
            logger.info(f'EarlyStopping Exceeded: No new best scores on iters {params.early_stopping_rounds}')
            break

    if logging_result:
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
    params = Parameters(**setup_params(vars(args), args.parameters))  # args，run_date，git_revisionなどを追加した辞書を取得

    # 結果出力用ディレクトリの作成
    result_dir = params.result_dir
    required_dirs = [result_dir, result_dir + "/train", result_dir + "/train/csv", result_dir + "/eval",
                    result_dir + "/test", result_dir + "/visualize", result_dir + "/visualize/csv",
                    result_dir+"/dataset"]
    make_dir(required_dirs)

    # パラメータを出力
    dump_params(params, f'{result_dir}')

    # ログ設定
    logger = logging.getLogger(__name__)
    set_logging(result_dir, file_name="train")  # ログを標準出力とファイルに出力するよう設定

    # train
    train_fe(params, logger)