# Python標準モジュール
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
from utils import dump_params, setup_params, set_logging, make_dir, fix_seed
from preprocess import preprocess
from models import cvae_with_fe
from graph_process import graph_utils


def setup_logging(params:'config.Parameters') -> Any:
    """tensorboardのSummaryWriterやcsv出力の初期設定を行う関数.

    Returns:
        Any: SummaryWriterインスタンス
    """
    # Tensorboardの設定
    writer = SummaryWriter(log_dir=f"{params.result_dir}")

    # epoch毎のlossが記載されるcsv
    for phase in ["train", "valid"]:
        with open(f"{params.result_dir}/train/csv/{phase}_loss_data.csv", "w") as f:
            f.write(f"epoch,loss,fe_loss,fe_rmse_loss\n")

    return writer


def train_cvae_with_fe(params: 'config.Parameters', logger: 'logging.Logger') -> float:
    """CVAE + feを学習する関数

        Args:
            params (config.Parameters)  : global変数のset
            logger (logging.Logger)	    : logging
    """

    # Seed値の固定
    fix_seed(params.seed)

    # deviceの設定
    device = torch.device(f"cuda:{params.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(params.gpu_id)

    # 学習結果のloggingなど
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

    # 前処理後のデータセットからグラフobjを復元する
    if params.logging_graph_vis_with_dataset_and_decoding:
        # train用のグラフをobjへ変換する
        train_dfs_code = torch.cat((train_label[0].unsqueeze(dim=2), train_label[1].unsqueeze(dim=2), train_label[2].unsqueeze(dim=2), train_label[3].unsqueeze(dim=2), train_label[4].unsqueeze(dim=2)), dim=2)
        train_graphs = []
        for code in train_dfs_code:
            graphs = graph_utils.dfs_code_to_graph_obj(
                code.cpu().detach().numpy(),
                [time_size, time_size, node_size, node_size, edge_size],
                edge_num=train_label[0].shape[1])
            train_graphs.append(graphs)
        # valid用のグラフをobjへ変換する
        valid_dfs_code = torch.cat((valid_label[0].unsqueeze(dim=2), valid_label[1].unsqueeze(dim=2), valid_label[2].unsqueeze(dim=2), valid_label[3].unsqueeze(dim=2), valid_label[4].unsqueeze(dim=2)), dim=2)
        valid_graphs = []
        for code in valid_dfs_code:
            graphs = graph_utils.dfs_code_to_graph_obj(
                code.cpu().detach().numpy(),
                [time_size, time_size, node_size, node_size, edge_size],
                edge_num=valid_label[0].shape[1])
            valid_graphs.append(graphs)
        # dictへ変換する
        dataset_graphs = {'train': train_graphs, 'valid': valid_graphs}

    # datasetを作成
    dfs_size = 2 * time_size + 2 * node_size + edge_size + conditional_size
    train_conditional = torch.cat([train_conditional for _ in range(train_dataset.shape[1])], dim=1).unsqueeze(2)
    valid_conditional = torch.cat([valid_conditional for _ in range(valid_dataset.shape[1])], dim=1).unsqueeze(2)
    train_dataset = torch.cat((train_dataset, train_conditional), dim=2)
    valid_dataset = torch.cat((valid_dataset, valid_conditional), dim=2)
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
        generator=g,									    # 乱数生成器を指定
        num_workers=0                                       # データloadingに使用するサブプロセスの数
    )
    valid_dl = DataLoader(
        TensorDataset(valid_label_args, valid_dataset),
        shuffle=False,
        batch_size=params.model_params["batch_size"],
        pin_memory=True,
        generator=g,
        num_workers=0
    )

    # 学習するモデルの定義
    model = cvae_with_fe.CVAEwithFE(dfs_size, time_size, node_size, edge_size, conditional_size, params, device)
    if params.args['fe_file']:
        model.fe.load_state_dict(torch.load(params.args["fe_file"], map_location=device))
    else:
        raise Exception("事前学習されたFEの重みdataへのPATHを指定してください.")

    if params.froze_fe_params:
        # FEモデルパラメータの更新を停止する
        for param in model.fe.parameters():
            param.requires_grad = False

    # (Optional) チェックポイントをload
    if params.args['checkpoint_file'] and params.args['init_epoch']:
        model.load_state_dict(torch.load(params.args['checkpoint_file'], map_location=device))
        init_epoch = int(params.args['init_epoch'])  # 初期エポック数
    else:
        init_epoch = 1

    # modelの概要をlogging
    model = model.to(device)  # modelをGPUに乗せる
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


    min_loss        = {"train": 1e10, "valid": 1e10}
    fe_min_loss    = {"train": 1e10, "valid": 1e10}
    best_epoch      = {"train": init_epoch, "valid": init_epoch}
    fe_best_epoch  = {"train": init_epoch, "valid": init_epoch}

    dataloader      = {"train": train_dl, "valid": valid_dl}                    # DataLoaderのdict
    label           = {"train": train_label, "valid": valid_label}              # ラベルのdict
    condition       = {"train": train_conditional, "valid": valid_conditional}  # Conditionのdict

    # 学習開始
    logger.info("Start training ...")
    for epoch in range(init_epoch, params.epochs + 1, 1):
        logger.info("Epoch: [%d/%d]:" % (epoch, params.epochs))

        # 各epochにlossを0で初期化する
        encoder_loss_per_epoch      = {"train": 0., "valid": 0.}
        five_tuples_dict            = {"tu": 0., "tv": 0., "lu": 0., "lv": 0., "le": 0.}
        decoder_loss_per_epoch_dict = {"train": five_tuples_dict.copy(), "valid": five_tuples_dict.copy()}
        decoder_acc_per_epoch_dict  = {"train": five_tuples_dict.copy(), "valid": five_tuples_dict.copy()}
        fe_loss_per_epoch          = {"train": 0., "valid": 0.}
        model_loss_per_epoch        = {"train": 0., "valid": 0.}

        # Open datasetのAPL, GraphTune出力のグラフのAPL, APLを出力するFE出力を記録する
        if params.logging_condition_by_graphtune_fe:
            # checkpointの時だけlogging
            if epoch % params.model_save_point == 0:
                make_dir([f"{params.result_dir}/train/csv/epoch_{epoch}"])
                with open(f"{params.result_dir}/train/csv/epoch_{epoch}/apl_output.csv", "w") as f:
                    f.write(f"iter,dataset_apl,graphtune_output_apl,fe_output\n")

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
                    mu, sigma, tu, tv, lu, lv, le, graph_property, generated_graph_labels, graph_obj_list = model(data)

                    # Logging condtition
                    if params.logging_condition_by_graphtune_fe:
                        if epoch % params.model_save_point == 0:
                            with open(f"{params.result_dir}/train/csv/epoch_{epoch}/apl_output.csv", "a") as f:
                                for j, g_i in enumerate(indicies):
                                    f.write(f"{j+i*params.model_params['batch_size']},{condition[phase][g_i][0][0].item()},{generated_graph_labels[j].item()},{graph_property[j][0][0].item()}\n")

                    # Save graph plot
                    if params.logging_graph_vis_with_dataset_and_decoding:
                        if epoch % params.model_save_point == 0:
                            target_dataset_graphs = [dataset_graphs[phase][gi.item()] for gi in indicies]
                            for j, (dataset_G, decode_G) in enumerate(zip(target_dataset_graphs, graph_obj_list)):
                                if j >= 2:
                                    break
                                fig, ax = plt.subplots(1, 2, figsize=(20, 10))
                                pos_1 = nx.spring_layout(dataset_G)
                                pos_2 = nx.spring_layout(decode_G)
                                nx.draw_networkx(dataset_G, pos_1, ax=ax[0])
                                nx.draw_networkx(decode_G, pos_2, ax=ax[1])
                                ax[0].set_title("Dataset")
                                ax[1].set_title("Decode (Argmax)")
                                fig.tight_layout()
                                plt.axis("off")
                                if phase == "train":
                                    plt.savefig(f"{params.result_dir}/train/csv/epoch_{epoch}/graph_{j+i*params.model_params['batch_size']}.png")
                                else:
                                    plt.savefig(f"{params.result_dir}/train/csv/epoch_{epoch}/graph_{j+i*params.model_params['batch_size']+train_label[0].shape[0]}.png")
                                plt.clf()
                                plt.close('all')

                    # Calculate loss
                    ## Encoder loss
                    encoder_loss = model.cvae.encoder.loss(mu, sigma)
                    encoder_loss_per_epoch[phase] += encoder_loss.item()

                    ## Decoder(Reconstruction) loss
                    results = {"tu": tu, "tv": tv, "lu": lu, "lv": lv, "le": le}
                    targets = {"tu": label[phase][0][indicies], "tv": label[phase][1][indicies], "lu": label[phase][2][indicies],
                            "lv": label[phase][3][indicies], "le": label[phase][4][indicies]}
                    decoder_loss_dict, decoder_total_loss = model.cvae.decoder.loss(results, targets)
                    for (key, val) in decoder_loss_dict.items():
                        decoder_loss_per_epoch_dict[phase][key] += val.item()

                    ## FE loss (A) : 元DatasetのconditionをFE lossのlabelとする
                    graph_property = torch.squeeze(graph_property, 1)
                    target_fe = condition[phase][indicies].to(device)
                    target_fe = target_fe.transpose(1, 0)[0]
                    fe_loss = model.fe.loss(graph_property, target_fe)
                    fe_loss_per_epoch[phase] += fe_loss.item()

                    ## FE loss (B) : GraphTune出力を確率分布へ変換し, Argmaxでグラフを確定, 確定されたグラフの特徴量をFE lossのlabelとする
                    ## loss計算できるようなdata形式に変換する
                    # graph_property = graph_property.squeeze(dim=1).squeeze(dim=1)
                    # ## lossを計算する
                    # fe_loss = model.fe.loss(graph_property, generated_graph_labels.to(device))
                    # fe_loss_per_epoch[phase] += fe_loss.item()

                    ## Model loss
                    ## model_lossは, 各lossのミニバッチのsumに, 対応するalphaなどの定数を乗じて, 和をとったものである.
                    model_loss = model.loss(encoder_loss, decoder_total_loss, fe_loss)
                    model_loss_per_epoch[phase] += model_loss.item()

                    # Calculate accuracy
                    acc_dict = model.cvae.decoder.accuracy(results, targets)
                    for (key, score) in acc_dict.items():
                        decoder_acc_per_epoch_dict[phase][key] += score

                    # 訓練の時だけ, 誤差逆伝搬 + 勾配クリッピング + オプティマイズする
                    if phase == "train":
                        model_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), params.model_params["clip_th"])
                        opt.step()


            # Save loss / acc
            ### encoder loss
            writer.add_scalar(f"{phase}_loss/encoder_loss", encoder_loss_per_epoch[phase] / data_num[phase], epoch)
            logger.info(f"    encoder_loss = {encoder_loss_per_epoch[phase] / data_num[phase]} [/graph]")
            ### decoder loss
            decoder_total_loss = 0
            for (key, val) in decoder_loss_per_epoch_dict[phase].items():
                decoder_total_loss += val / data_num[phase]
                writer.add_scalar(f"{phase}_loss/{key}_loss", val / data_num[phase], epoch)
                logger.info(f"    {key}_loss = {val / data_num[phase]} [/graph]")
            logger.info(f"    decoder_total_loss = {decoder_total_loss} [/graph]")
            ### decoder acc
            for (key, val) in decoder_acc_per_epoch_dict[phase].items():
                writer.add_scalar(f"{phase}_acc/{key}_acc",
                                val / math.ceil(data_num[phase] / params.model_params['batch_size']), epoch)
                logger.info(
                    f"    {key}_acc = {val / math.ceil(data_num[phase] / params.model_params['batch_size'])}")
            ### fe loss
            writer.add_scalar(f"{phase}_loss/fe_loss", fe_loss_per_epoch[phase] / data_num[phase], epoch)
            writer.add_scalar(f"{phase}_loss/fe_rmse_loss", math.sqrt(fe_loss_per_epoch[phase] / data_num[phase]), epoch)
            logger.info(f"    fe_loss = {fe_loss_per_epoch[phase] / data_num[phase]}")
            logger.info(f"    fe_rmse_loss = {math.sqrt(fe_loss_per_epoch[phase] / data_num[phase])}")
            ### model loss
            writer.add_scalar(f"{phase}_loss/model_loss", model_loss_per_epoch[phase] / data_num[phase], epoch)
            logger.info(f"    model_loss = {model_loss_per_epoch[phase] / data_num[phase]}")
            with open(f"{params.result_dir}/train/csv/{phase}_loss_data.csv", "a") as f:
                f.write(f"{epoch},{model_loss_per_epoch[phase] / data_num[phase]},{fe_loss_per_epoch[phase] / data_num[phase]},{math.sqrt(fe_loss_per_epoch[phase] / data_num[phase])}\n")


        # Save model at checkpoint
        if epoch % params.model_save_point == 0:
            torch.save(model.state_dict(), params.result_dir + "/train/weight_" + str(epoch))
            logger.info(f'Checkpoint: {epoch}')

        # Save best model
        for phase in ["train", "valid"]:
            # best CVAE loss
            if model_loss_per_epoch[phase] < min_loss[phase]:
                best_epoch[phase]   = epoch
                min_loss[phase]     = model_loss_per_epoch[phase]
                torch.save(model.state_dict(), params.result_dir + "/train/" + phase + "_best_weight")
                logger.info(f'  Update {phase} best epoch: {epoch}')
            # best FE loss
            if fe_loss_per_epoch[phase] < fe_min_loss[phase]:
                fe_best_epoch[phase]   = epoch
                fe_min_loss[phase]     = fe_loss_per_epoch[phase]
                torch.save(model.state_dict(), params.result_dir + "/train/fe_" + phase + "_best_weight")
                logger.info(f'  Update fe {phase} best epoch: {epoch}')

    writer.close()
    for phase in ["train", "valid"]:
        logger.info(f"{phase} best epoch : {best_epoch[phase]}")
        logger.info(f"FE {phase} best epoch : {fe_best_epoch[phase]}")
    logger.info("train complete!")



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

    # グローバル変数をlog出力
    logger.info('parameters: ')
    logger.info(params)

    # train
    train_cvae_with_fe(params=params, logger=logger)