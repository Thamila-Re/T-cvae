"""便利な関数群"""
from __future__ import annotations  # Python 3.7, 3.8はこの記述が必要
import torch
from torch.distributions import Categorical
import subprocess
from logging import getLogger, StreamHandler, DEBUG, Logger, INFO, Formatter, FileHandler
import json
from datetime import datetime
import os
from dataclasses import asdict
from typing import Any
import glob
import numpy as np
import pandas as pd
import random
import shutil
import pickle
import joblib


def get_git_revision() -> str:
    """
    現在のGitのリビジョンを取得

        Returns:
             str: revision ID
    """
    cmd = "git rev-parse HEAD"
    revision = subprocess.check_output(cmd.split())  # 現在のコードのgitのリビジョンを取得
    return revision.decode()


def setup_params(args_dict: dict[str, Any], path: str = None, update_params = None) -> dict[str, Any]:
    """
    コマンドライン引数などの辞書を受け取り，実行時刻，Gitのリビジョン，jsonファイルからの引数と結合した辞書を返す．
    
        Args:
            args_dict (dict): argparseのコマンドライン引数などから受け取る辞書
            path (str, optional): パラメータが記述されたjsonファイルのパス
            update_params (dict, optional): jsonファイルのパラメータを上書きしたい場合に指定する辞書
        Returns:
            dict: args_dictと実行時刻，Gitのリビジョン，jsonファイルからの引数が結合された辞書．
                構造は {'args': args_dict, 'git_revision': <revision ID>, 'run_date': <実行時刻>, ...}．
    """
    run_date = datetime.now()
    git_revision = get_git_revision()  # Gitのリビジョンを取得

    param_dict = {}
    if path:
        param_dict = json.load(open(path, 'r'))  # jsonからパラメータを取得
    if "run_date" not in param_dict:
        param_dict.update({'run_date': run_date.strftime('%Y%m%d_%H%M%S')})  # 実行時刻を上書き
    if "result_dir" not in param_dict or param_dict["result_dir"] == '':
        param_dict.update({'result_dir': f'result/{param_dict["run_date"]}'}) # 出力ディレクトリパスを上書き
    param_dict.update({'args': args_dict})  # コマンドライン引数を上書き
    param_dict.update({'git_revision': git_revision})  # Gitリビジョンを上書き
    if args_dict['result_dir'] is not None:
        param_dict.update({'result_dir': args_dict['result_dir']})
    if update_params:
        param_dict.update(update_params)
    return param_dict


def dump_params(params: 'config.Parameters', outdir: str, partial: bool = False, file_name:str = 'parameters.json') -> None:
    """
    データクラスで定義されたパラメータをjson出力する関数
    
    Args:
        params (:ogj: `Parameters`): パラメータを格納したデータクラス
        outdir (str): 出力先のディレクトリ
        partial (bool, optional): Trueの場合，args，run_date，git_revision を出力しない，
        file_name (str, optional): 出力ファイル名
    """
    params_dict = asdict(params)  # デフォルトパラメータを取得
    if os.path.exists(f'{outdir}/{file_name}'):
        raise Exception(f'"{file_name}" is already exist. ')
    if partial:
        del params_dict['args']  # jsonからし指定しないキーを削除
        del params_dict['run_date']  # jsonからし指定しないキーを削
        del params_dict['git_revision']  # jsonからし指定しないキーを削
    with open(f'{outdir}/{file_name}', 'w') as f:
        json.dump(params_dict, f, indent=4)  # デフォルト設定をファイル出力


def set_logging(result_dir: str, file_name: str = "log", 
                stdout_log_level: str ="INFO", fileout_log_level: str ="DEBUG") -> Logger:
    """
    ログを標準出力とファイルに書き出すよう設定する関数．

        Args:
            result_dir (str): ログの出力先
            stdout_log_level: str = "INFO": 標準出力(ターミナル)のログレベル
            fileout_log_level: str = "DEBUG": ファイル出力のログレベル
        Returns:
            設定済みのrootのlogger
    
        Example: 
        >>> # logger = logging.getLogger(__name__)
        >>> # set_logging(result_dir)
        >>> # logger.info('log message...')
    """
    logger = getLogger()
    # 実際に出力されるのレベルは、handlerとfile_handlerでそれぞれ指定するので、
    # 以下では、最も出力レベルが低いDEBUGにしておく。
    logger.setLevel(DEBUG)  # 全体のログレベル
    
    # formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # ログのフォーマット
    formatter = Formatter('%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s')  # ログのフォーマット

    # 標準出力へのログ出力設定
    handler = StreamHandler()
    handler.setLevel(stdout_log_level.upper()) # ターミナル出力ログレベル
    handler.setFormatter(formatter)  # フォーマットを指定
    logger.addHandler(handler)
    
    # ファイル出力へのログ出力設定
    file_handler = FileHandler(f'{result_dir}/{file_name}.log', 'w')  # ログ出力ファイル
    file_handler.setLevel(fileout_log_level.upper()) # ファイル出力ログレベル
    file_handler.setFormatter(formatter)  # フォーマットを指定
    logger.addHandler(file_handler)

def update_json(json_file: str, input_dict: dict[str, Any]) -> None:
    """jsonファイルをupdateするプログラム
        import json が必要

        Args:
            json_file (str): jsonファイルのpath
            input_dict (dict): 追加もしくは更新したいdict
    """
    with open(json_file) as f:
        df = json.load(f)

    df.update(input_dict)

    with open(json_file, 'w') as f:
        json.dump(df, f, indent=4)

def try_gpu(device, obj):
    """objectを指定されたdeviceに乗せる.

    Args:
        device (): device info
        obj (any): any object

    Returns:
        (any): 指定されたdeviceに乗ったobject
    """
    import torch
    return obj.to(device)
    # if torch.cuda.is_available():
    #     return obj.cuda(device)
    # return obj

def get_gpu_info(nvidia_smi_path: str = 'nvidia-smi', no_units: bool = True) -> str:
    """
    空いているgpuの番号を持ってくるプログラム

        Returns:
            str: 空いているgpu番号 or 'cpu'
    """
    keys = (
        'index',
        'uuid',
        'name',
        'timestamp',
        'memory.total',
        'memory.free',
        'memory.used',
        'utilization.gpu',
        'utilization.memory'
    )
    if torch.cuda.is_available():
        nu_opt = '' if not no_units else ',nounits'
        cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
        output = subprocess.check_output(cmd, shell=True)
        lines = output.decode().split('\n')
        lines = [line.strip() for line in lines if line.strip() != '']

        gpu_info = [{k: v for k, v in zip(keys, line.split(', '))} for line in lines]

        min_gpu_index = 0
        min_gpu_memory_used = 100
        for gpu in gpu_info:
            gpu_index = gpu['index']
            gpu_memory = int(gpu['utilization.gpu'])
            if min_gpu_memory_used >= gpu_memory:
                min_gpu_memory_used = gpu_memory
                min_gpu_index = int(gpu_index)

        return "cuda:" + str(min_gpu_index)
    else:
        return 'cpu'

def make_dir(required_dirs) :
    """入力されたディレクトリパスのリストから、ディレクトリを作成する関数

    既に存在しているディレクトリパスが指定された場合は、スキップされる.

        Args:
            required_dirs (list) : 作成したいディレクトリパスのリスト
    """
    dirs = glob.glob("*")
    for required_dir in required_dirs:
        if not os.path.isdir(required_dir):
            print("generate file in current dir...")
            print("+ "+required_dir)
            os.makedirs(required_dir)
        print("\n")

def convert2onehot(vec, dim):
    """
    特徴量のnumpy配列をonehotベクトルに変換
    
    Args:
        vec (): 特徴量のnumpy行列, int型 (サンプル数分の1次元行列)．
        dim (int): onehot vectorの次元
    
    Returns:
        (torch.Tensor) : onehot vectorのtensor行列
    """
    import torch
    return torch.Tensor(np.identity(dim)[vec])

def padding(vecs, flow_len, value=0):
    """flowの長さを最大flow長に合わせるためにzero padding
    
    Args:
        vecs (): flow数分のリスト. リストの各要素はflow長*特徴量長の二次元numpy配列
        flow_len (int): flow長
        value (int): paddingするvectorの要素値
    
    Returns:
        (): データ数*最大flow長*特徴量長の3次元配列
    """
    for i in range(len(vecs)):
        flow = vecs[i]
        if len(flow.shape)==2:
            diff_vec = np.ones((flow_len-flow.shape[0], flow.shape[1]))
        elif len(flow.shape) == 3:
            diff_vec = np.ones((flow_len-flow.shape[0], flow.shape[1], flow.shape[2]))
        else:
            diff_vec = np.ones(flow_len-flow.shape[0])
        diff_vec *= value
        vecs[i] = np.concatenate((flow, diff_vec), 0)
    return np.array(vecs)

def sample_dist(dist_tensor: 'torch.Tensor') -> tuple[torch.Tensor, torch.Tensor]:
    """テンソルで表現された分布からサンプリングする
    
    Parameters
    ----------
    dist_tensor: torch.Tensor
        カテゴリカルな確率分布を表すテンソル (和が1である必要はない)
    Returns
    -------
        サンプリングした結果のカテゴリと何番目に高い確率が得られたかを示すint型のタプル
        
    Examples
    -------
        >>> dist = torch.Tensor([[[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]]])
        >>> sample_dist(dist)[0]
        (tensor([[1], 
                [0]]), tensor([[0], 
                [0]]))
    """
    categorical_obj = Categorical(dist_tensor)
    sample = categorical_obj.sample()
    sorted_values, sorted_indices = torch.sort(dist_tensor, descending=True)
    selected_rank = torch.nonzero(sorted_indices.squeeze() == sample)[:,1]
    selected_rank = selected_rank.unsqueeze(dim=-1)
    return sample,selected_rank

def sample_topk_topp(dist_tensor: 'torch.Tensor', k: int, p: float, mode='both') -> tuple[torch.Tensor, torch.Tensor]:
    """確率分布の上位の確率k個を選択し、その確率から新たな分布を生成する。

    Parameters
    ----------
    dist_tensor: torch.Tensor
        カテゴリカルな確率分布を表すテンソル (和が1である必要はない)
    Returns
    -------
        新たに生成した分布からサンプリングした結果のカテゴリ
        
    Examples
    -------
        >>> dist = torch.Tensor([[[0.2, 0.5, 0.1, 0.05, 0.0]], [[0.0, 0.35, 0.5, 0.15, 0.0]]])
        >>> sample_topk_topp(dist,6,0.4)
        (tensor([[1], 
                [2]]), tensor([[0], 
                [0]]))
    """
    # kやpが適切な範囲内でないときの調整
    # pが適切でないときは実質argmaxになる
    if dist_tensor.size()[-1] < k:
        k = dist_tensor.size()[-1]
    if torch.max(dist_tensor) > p:
        p = torch.max(dist_tensor).item()

    sorted_dist, sorted_indices = torch.sort(dist_tensor, descending=True)
    if (mode=='both' or mode=='topk'):
        sorted_dist, sorted_indices = torch.topk(dist_tensor, k)
    if (mode=='both' or mode=='topp'):
        cumulative_probs = torch.cumsum(sorted_dist, dim=-1)
        top_p_filter = cumulative_probs <= p
        sorted_dist = sorted_dist * top_p_filter.float()

    topk_sampled,selected_rank = sample_dist(sorted_dist)
    sample = torch.gather(sorted_indices.squeeze(dim=1), 1, topk_sampled)
    return sample,selected_rank

def get_condition_values(condition_params, condition_values, label_nums=3):
    """

    Args:
        condition_params (): _description_
        condition_values (): _description_
        label_nums (int, optional): _description_. Defaults to 3.

    Returns:
        (): _description_
    """
    labels = [[] for i in range(label_nums)]
    keys = condition_params
    for key in keys:
        values = condition_values[key]
        for i in range(label_nums):
            labels[i].append(values[i])

    labels = [np.prod(i) for i in labels]
    return labels

def concat_csv(csv_paths):
    """複数のcsvファイルを結合する関数

    Parameters
    ----------
    csv_paths : list
        結合したいcsvファイルのパスのリスト

    Returns
    -------
    pandas.df
        csvファイルを結合してtypeを追加したpandasのデータフレーム
    """
    df_concat = pd.read_csv(csv_paths[0])
    df_concat['type'] = os.path.splitext(os.path.basename(csv_paths[0]))[0]

    for path in csv_paths[1:]:
        df_add = pd.read_csv(path)
        df_add['type'] = os.path.splitext(os.path.basename(path))[0]
        df_concat = pd.concat([df_concat,df_add])

    return df_concat

def fix_seed(seed: int):
    """再現性のために、Seed値を固定する関数.

    Args:
        seed (int): seed値
    """
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # GPUの処理を決定的にする
    # torch.use_deterministic_algorithms = True   # PyTorch操作で決定論的アルゴリズムを使用する
                                                # 利用可能な決定論的アルゴリズムがない場合はError
    # torch.use_deterministic_algorithms(True) # おそらくこれが正しい書き方                              
    torch.backends.cudnn.benchmark = False      # 畳み込み演算の箇所の再現性

def seed_worker(worker_id):
    '''DataLoaderのworkerのseedを設定する関数.
        参考 : https://qiita.com/north_redwing/items/1e153139125d37829d2d 
        コピペなので、引数の意味は不明
    '''
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def copy_dirs(old_paths: list, new_path: str):
    """指定したディレクトリ群を別の新たなディレクトリ内にコピーしてまとめる関数.

    Args:
        old_paths (list): コピー元のディレクトリのパスのリスト(末尾に/を付ける)
        new_path   (str): コピー先のディレクトリのパス(末尾に/を付ける)
    """
    for old_path in old_paths:
        shutil.copytree(old_path.rstrip(), new_path + old_path.split("/")[1] + "/" + old_path.split("/")[3])

def graphgen_evaluate(value):
    graph_files = glob.glob(f"result/graphgen_{str(value)}/eval/"r"*")
    graphs = []
    for file in graph_files:
        with open(file, 'rb') as f:
            graphs.append(pickle.load(f))

    joblib.dump(graphs, f'result/graphgen/eval/Length_{str(value)}')

def weighted_mean_std(values: pd.Series, weights: pd.Series) -> tuple[int, int]:
    """加重平均を行い、その値と標準偏差を返す関数.

    Args:
        values  (pd.Series): 平均を取りたい値のSeries
        weights (pd.Series): valuesに対応する重み
    Returns
        w_mean, w_std (tuple[int, int]): (平均, 標準偏差)
    """
    totals = values*weights
    w_mean = totals.sum()/weights.sum()
    squared_dev = lambda v: (v-w_mean)**2
    w_std = ((values.agg(squared_dev)*weights).sum()/weights.sum())**0.5
    return w_mean, w_std

if __name__=='__main__':
    print(get_gpu_info())
    # graphgen_evaluate(5.0)
