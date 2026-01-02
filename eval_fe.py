"""
FEの推定精度を計測するモジュール.
"""
import argparse
import logging
import math

import joblib
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchinfo import summary

from config import common_args, Parameters
from models import fe
from utils import setup_params
from utils import set_logging, fix_seed


def eval_fe(params:'config.Parameters', logger:'logging.Logger'=None) -> None:

    # Deviceはcpuに限定
    device = "cpu"

    # 結果dirの名前を取得
    run_time = params.args['eval_model'].split('/')[1]

    # eval用dirのpath
    eval_dir = "result/" + run_time + "/eval/"

    # Load preprocessed dataset
    train_dataset       = joblib.load(f"./result/{run_time}/dataset/train/onehot")
    train_label         = joblib.load(f"./result/{run_time}/dataset/train/label")
    train_conditional   = joblib.load(f"./result/{run_time}/dataset/train/conditional")
    valid_dataset       = joblib.load(f"./result/{run_time}/dataset/valid/onehot")
    valid_label         = joblib.load(f"./result/{run_time}/dataset/valid/label")
    valid_conditional   = joblib.load(f"./result/{run_time}/dataset/valid/conditional")
    time_size, node_size, edge_size, conditional_size = joblib.load(f"./result/{run_time}/dataset/param")

    # calc DFS code size
    dfs_size = 2 * time_size + 2 * node_size + edge_size + conditional_size

    # logging size
    logger.info("--------------")
    logger.info(f"time size: {time_size}")
    logger.info(f"node size: {node_size}")
    logger.info(f"edge size: {edge_size}")
    logger.info(f" dfs size: {dfs_size}")
    logger.info(f"conditional size: {conditional_size}")
    logger.info("--------------")

    # labelをdeviceに乗せる
    train_label = [element.to(device) for element in train_label]
    valid_label = [element.to(device) for element in valid_label]

    # 次元を調整
    train_conditional = torch.cat([train_conditional for _ in range(train_dataset.shape[1])], dim=1).unsqueeze(2)
    valid_conditional = torch.cat([valid_conditional for _ in range(valid_dataset.shape[1])], dim=1).unsqueeze(2)

    # Dataloaderのseed固定
    g = torch.Generator()
    g.manual_seed(params.seed)

    # dataloaderを作成
    data_num = {"train": train_dataset.shape[0], "valid": valid_dataset.shape[0]}
    train_label_args = torch.LongTensor(list(range(data_num["train"])))
    valid_label_args = torch.LongTensor(list(range(data_num["valid"])))
    train_dl = DataLoader(
        TensorDataset(train_label_args, train_dataset),
        shuffle=False,										# evalなので、shuffleしない
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

    # 生成されたグラフが十分なサイズであるか判別する関数
    is_sufficient_size = lambda graph: True if graph.number_of_nodes() > params.size_th else False

    # modelを定義
    model = fe.FeatureEstimator(dfs_size - conditional_size, params, device)
    logger.info("モデル概要")
    logger.info(
        summary(
            model,
            input_size=(params.model_params["batch_size"], train_dataset.shape[1], train_dataset.shape[2]),
            col_names=["output_size", "num_params"],
            device=device
        )
    )

    # 評価対象のモデルをload
    model.load_state_dict(torch.load(params.args["eval_model"], map_location=device))
    model.eval()

    # eval
    dataloader = {"train": train_dl, "valid": valid_dl}  # DataLoaderのdict
    condition = {"train": train_conditional, "valid": valid_conditional}    # Conditionのdict
    model_loss_per_epoch = {"train": 0., "valid": 0.}

    for phase in ["train", "valid"]:
        logger.info(f"  {phase}:")

        # モデル出力値と正解値をグラフ毎に書き込むcsv
        # with open(f"{eval_dir}{phase}_pred_data.csv", "w") as f:
        with open(f"{eval_dir}{phase}_pred_data_ER_1000.csv", "w") as f:
            f.write(f"epoch,index,pred,correct,abs(pred-correct)\n")

        # データをイレテー卜する
        for i, (indicies, data) in enumerate(dataloader[phase], 0):
            data = data.to(device)
            # predict
            graph_property = model(data)

            # MSE loss
            ## model_lossは, 各lossのミニバッチのsumに, 対応するalphaなどの定数を乗じて, 和をとったものである.
            graph_property = torch.squeeze(graph_property, 1)
            target_fe = condition[phase][indicies].to(device)
            target_fe = target_fe.transpose(1, 0)[0]
            model_loss = model.loss(graph_property, target_fe)
            model_loss_per_epoch[phase] += model_loss.item()

            # logging output, correct
            # with open(f"{eval_dir}{phase}_pred_data.csv", "a") as f:
            with open(f"{eval_dir}{phase}_pred_data_ER_1000.csv", "w") as f:
                for j, (pred, correct) in enumerate(zip(graph_property, target_fe)):
                    f.write(f"1,{indicies[j].item()},{pred.item()},{correct.item()},{torch.abs(pred-correct).item()}\n")

        # Save info
        logger.info(f"    model_loss = {model_loss_per_epoch[phase] / data_num[phase]} [hop^2 / graph]")
        logger.info(f"    model_rmse_loss = {math.sqrt(model_loss_per_epoch[phase] / data_num[phase])} [hop / graph]")

    logger.info("eval complete!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = common_args(parser)
    args = parser.parse_args()
    params = Parameters(**setup_params(vars(args), args.parameters))  # args，run_date，git_revisionなどを追加した辞書を取得

    if params.args['eval_model'] is None:
        raise Exception("評価対象のモデルを指定してください.")

    # ログ設定
    logger = logging.getLogger(__name__)
    result_dir = params.args['eval_model'].split('/')[0] + '/' + params.args['eval_model'].split('/')[1]
    set_logging(result_dir, file_name="eval_ER_1000")

    # seed値を固定
    print(f"Seedを{params.seed}に固定します。")
    fix_seed(params.seed)

    # eval
    eval_fe(params, logger)
