"""ハイパーパラメータのチューニング
"""
# Python標準モジュール
import argparse
import copy
import joblib
import logging
import shutil
# Third-partyモジュール
import numpy as np
import optuna
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import yaml
# 自作モジュール
from config import common_args, Parameters
from models import fe
from preprocess import preprocess
import train_fe
from utils import dump_params, setup_params, fix_seed
from utils import set_logging, make_dir



def tune_fe(params: 'config.Parameters', logger: 'logging.Logger') -> None:
    """Tuning FE hyper parameters

    Args:
        params (config.Parameters)  : global変数のset
        logger (logging.Logger)	    : logging
    """

    # Seed値の固定
    fix_seed(params.seed)

    # setting for device
    device = torch.device(f"cuda:{params.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(params.gpu_id)

    # 結果出力用dir
    result_dir = params.result_dir

    # 前処理が必要なら、ここで１度だけ行う
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
        ## 前処理完了後はtrainで再び前処理しないように、flagをFalseにする
        params.args['preprocess'] = False
        args.preprocess = False


    # 最適化対象の目的オブジェクト
    def tuning_trial(trial: optuna.Trial, result_dir:str) -> float:
        # チューニング対象のパラメータの設定
        global params
        params.model_params["lr"]                  = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        params.model_params["fe_emb_size"]        = trial.suggest_int("fe_emb_size", 256, 1024, step=128)
        params.model_params["fe_hidden_size"]     = trial.suggest_int("fe_hidden_size", 128, 512, step=128)
        params.model_params["fe_lstm_num_layer"]  = trial.suggest_int("fe_lstm_num_layer", 1, 3)
        params.model_params["fe_dropout"]         = trial.suggest_float("fe_dropout", 0.0, 1.0)
        # 前処理・データ整形
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
        # Dataloaderのseed固定
        g = torch.Generator()
        g.manual_seed(params.seed)
        # Dataloader
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
        # model
        model = fe.FeatureEstimator(dfs_size - 1, params, device)
        model = model.to(device)  # modelをGPUに乗せる
        # optimizer
        opt = optim.Adam(model.parameters(), lr=params.model_params["lr"])
        # train
        valid_min_loss      = 1e10  # validでのmodel_lossの最小値
        valid_best_epoch    = 1     # validでの最もmodel_lossが小さい時のepoch
        early_stopping_cnt  = 0     # epochカウント. valid lossが更新されたら, 0にリセットされる.
        dataloader  = {"train": train_dl, "valid": valid_dl}                    # DataLoaderのdict
        label       = {"train": train_label, "valid": valid_label}              # ラベルのdict
        condition   = {"train": train_conditional, "valid": valid_conditional}  # Conditionのdict
        for epoch in range(1, params.epochs + 1, 1):
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

                # Save loss / acc
                ### model loss
                logger.info(f"    model_loss = {model_loss_per_epoch[phase] / data_num[phase]}")

            # Save valid best model
            if model_loss_per_epoch["valid"] < valid_min_loss:
                early_stopping_cnt = 0
                valid_best_epoch = epoch
                valid_min_loss = model_loss_per_epoch["valid"]
                logger.info(f'  Update valid best epoch: {epoch}')
            else:
                early_stopping_cnt += 1

            # early stopping
            if early_stopping_cnt >= params.early_stopping_rounds:
                logger.info(f'EarlyStopping Exceeded: No new best scores on iters {params.early_stopping_rounds}')
                break

        return valid_min_loss   # tune用にvalidの最小lossを返す


    # studyインスタンスを作成
    if params.args['db_path'] is None:
        ## db pathが指定されてないなら、実行時間のdir直下にdbファイルを作成
        db_path = f"sqlite:///{params.result_dir}/fe_tune_apl_uniform.db"
    else:
        ## db pathが指定されているなら、そのdbを参照
        db_path = params.args['db_path']
    study = optuna.create_study(
                study_name="fe_tune_apl_uniform",      # study name
                storage=db_path,            # database URL
                load_if_exists=True,        # study nameが既存なら、それをloadする
                direction='minimize'        # 最適化の方向
            )
    # Optunaによる最適化
    study.optimize(lambda trial: tuning_trial(trial, result_dir=result_dir), n_trials=params.opt_epoch)
    # study.optimize(tuning_trial(result_dir=result_dir), n_trials=params.opt_epoch)
    logger.info("--------------------------")
    logger.info(study.best_params)
    logger.info("--------------------------")

    # Hyper parametersをyaml形式で保存
    with open(params.result_dir + "/fe_best_tune.yml", "w") as f:
        f.write(yaml.dump(study.best_params))
    logger.info("tune complete!")



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
    set_logging(result_dir, file_name="tune")  # ログを標準出力とファイルに出力するよう設定

    # グローバル変数をlog出力
    logger.info('parameters: ')
    logger.info(params)

    # tune
    tune_fe(params=params, logger=logger)