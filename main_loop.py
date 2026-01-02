import os
import logging
from datetime import datetime
import argparse
from config import common_args, Parameters
from utils import dump_params, setup_params
from utils import set_logging, make_dir
from utils import fix_seed
from train import train_cvae
from eval import eval
from visualize import graph_plot

first_run_date = datetime.now().strftime('%Y%m%d_%H%M%S')
condition_values_apl = [
    [3.0, 4.0, 5.0],
    [4.0, 5.0, 6.0],
    [5.0, 6.0, 7.0],
    [6.0, 7.0, 8.0],
    [7.0, 8.0, 9.0],
    [8.0, 9.0, 10.0]
]
result_dir_list = []

# コマンドライン引数の設定
parser = argparse.ArgumentParser()
parser = common_args(parser)  # コマンドライン引数引数を読み込み
# parser.add_argument("--main")  # 実行スクリプト固有のコマンドライン引数があればここに記入する．
args = parser.parse_args() # 実行時は train, eval, visualize のコマンドライン引数を全て指定

params = Parameters()
cond_copy = params.condition_values # condition_values の初期値を取得

for i in range(6):
    # args_dict = setup_params(vars(args))
    args_dict = setup_params(vars(args), args.parameters)

    # 結果出力用ディレクトリの作成
    result_dir = f'result/{first_run_date}/{args_dict["run_date"]}'
    required_dirs = [result_dir, result_dir + "/train", result_dir + "/train/csv", result_dir + "/eval",
                    result_dir + "/test", result_dir + "/visualize", result_dir + "/visualize/csv",
                    result_dir+"/dataset"]
    make_dir(required_dirs)

    args_dict["gpu_id"] = 0

    eval_epochs = 3000
    run_epochs = 10000

    args_dict.update({"result_dir": result_dir})
    # args_dict["args"]["eval_model"] = f'{result_dir}/train/valid_best_weight'
    args_dict["args"]["eval_model"] = f'{result_dir}/train/weight_{eval_epochs}'
    args_dict["args"]["eval_graphs"] = f'{result_dir}/eval/'

    result_dir_list.append(result_dir)

    args_dict.update({"condition_params": ["Average path length"]})
    cond_copy["Average path length"] = condition_values_apl[i]
    args_dict.update({"condition_values": cond_copy})

    if i > 0:
        w1000_path = f'{result_dir_list[i-1]}/train/weight_{eval_epochs}' 
        best_path = f'{result_dir_list[i-1]}/train/valid_best_weight'
        if os.path.exists(w1000_path):
            args_dict["args"]["checkpoint_file"] = w1000_path
        else:
            args_dict["args"]["checkpoint_file"] = best_path
        args_dict["args"]["init_epoch"] = 1
    
    if i == 1:
        args_dict.update({
            "input_graph_path": [
                result_dir_list[0] + "/eval/" + "Average path length" + "_" + str(condition_values_apl[0][2]) + ".pkl"
            ],
            "input_original_graph_num": 1000
        })
    elif i >= 2:
        args_dict.update({
            "input_graph_path": [
                result_dir_list[i-1] + "/eval/" + "Average path length" + "_" + str(condition_values_apl[i-1][2]) + ".pkl",
                result_dir_list[i-2] + "/eval/" + "Average path length" + "_" + str(condition_values_apl[i-2][2]) + ".pkl"
            ],
            "input_original_graph_num": 0
        })
    
    args_dict.update({"epochs": run_epochs}) # debug

    params = Parameters(**args_dict)  # args，run_date，git_revisionなどを追加した辞書を取得

    # 以下，train.py を参照
    
    # パラメータを出力
    dump_params(params, result_dir)

    # ログ設定
    # logger = logging.getLogger(__name__)
    logger = set_logging(result_dir=result_dir, logger_name=__name__)  # ログを標準出力とファイルに出力するよう設定

    # グローバル変数をlog出力
    logger.info('parameters: ')
    logger.info(params)

    # train
    print(train_cvae(params=params, logger=logger))


    # eval.py 参照

    # if params.args['eval_model'] is None:
    #     print("評価対象のモデルを指定してください.")
    #     exit()

    # seedの固定
    seed = params.seed
    print("Seedを%dに固定します。"%seed)
    fix_seed(seed)
    eval(params)

    # visualize
    graph_plot(params)


    # logger のハンドラ削除
    for h in logger.handlers:
        logger.removeHandler(h)
