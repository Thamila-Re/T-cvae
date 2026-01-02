"""
Conditional Variational AutoEncoder(CVAE) + Feature estimatorモデルのインターフェースを定義するモジュール.
"""

import torch
from torch import nn
import sys
import os
import torch.nn.functional as F
import numpy as np
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import try_gpu, sample_dist, convert2onehot
from models.fe import FeatureEstimator
from models.cvae import CVAE
from graph_process import graph_utils, graph_statistic


class CVAEwithFE(nn.Module):
    """Conditional VAE + Feature estimatorのインターフェース

    input_data => CVAE(Encoder, Decoder) => output_data => FeatureEstimator() => predicted_graph_property
    """
    def __init__(self,
                dfs_size:int, time_size:int, node_size:int, edge_size:int, condition_size:int,
                params:'config.Parameters', device:str) -> None:
        super(CVAEwithFE, self).__init__()
        self.gamma = params.model_params['gamma']
        self.flag_softmax = params.args["softmax"]
        self.softmax = nn.Softmax(dim=2)
        self.fe_preprocess_onehot = params.fe_preprocess_onehot
        self.fe_preprocess_convert_graph_obj = params.fe_preprocess_convert_graph_obj
        self.condition_params = params.condition_params
        self.condition_round  = params.condition_round
        self.time_size = time_size
        self.node_size = node_size
        self.edge_size = edge_size
        self.device = device
        self.cvae = CVAE(dfs_size, time_size, node_size, edge_size, condition_size, params, device)
        self.fe = FeatureEstimator(dfs_size-1, params, device)

    def forward(self, x):
        """順伝播

        Args:
            x (torch.Tensor): 入力データ(shape = [batch_size, sequense_size, feature_size])

        Returns:
            (torch.Tensor): Encoder出力で、平均に相当する値
            (torch.Tensor): Encoder出力で、分散に相当する値
            (torch.Tensor): timestamp uに相当するDecoder出力
            (torch.Tensor): timestamp vに相当するDecoder出力
            (torch.Tensor): node label uに相当するDecoder出力
            (torch.Tensor): node label vに相当するDecoder出力
            (torch.Tensor): edge label eに相当するDecoder出力
            (torch.Tensor): FE出力で、指定されたグラフ特徴量に相当する値
            (torch.Tensor): Decoder出力の分布からArgmaxをとることで得られるグラフの指定された特徴量
        """
        # CVAE
        mu, sigma, tu, tv, lu, lv, le = self.cvae(x)

        # FEのための前処理
        pretreated_dfs_code, generated_labels, generated_graph = self.fe_preprocess(
                        tu_copy=tu.clone(),
                        tv_copy=tv.clone(),
                        lu_copy=lu.clone(),
                        lv_copy=lv.clone(),
                        le_copy=le.clone())

        # FE
        graph_property = self.fe(pretreated_dfs_code.to(torch.float32))

        return mu, sigma, tu, tv, lu, lv, le, graph_property, generated_labels, generated_graph

    def loss(self, encoder_loss, decoder_loss, fe_loss):
        """Loss function

        Args:
            encoder_loss (_type_): Encoder modelのloss
            decoder_loss (_type_): Decoder modelのloss
            fe_loss (_type_): FE modelのloss

        Returns:
            (): CVAEwithFEのloss
        """
        cvae_loss = self.cvae.loss(encoder_loss, decoder_loss) + self.gamma * fe_loss
        return cvae_loss

    def generate(self, data_num:int, conditional_label:'torch.Tensor', max_size:int,
                z:'torch.Tensor'=None, is_output_sampling:bool=True) -> 'torch.Tensor':
        """Generate graph samples

        Args:
            data_num                   (int): 生成サンプル数
            conditional_label (torch.Tensor): 条件として与えるラベル情報
            max_size                   (int): 最大エッジ数
            z                 (torch.Tensor): 潜在空間からサンプリングされたデータ
            is_output_sampling        (bool): Trueなら返り値を予測dfsコードからargmaxしたものに. Falseなら予測分布を返す

        Returns:
            (torch.Tensor): 生成されたサンプルの5-tuplesの各要素のデータ
        """
        tu, tv, lu, lv, le = self.cvae.generate(data_num, conditional_label, max_size, z, is_output_sampling)
        return tu, tv, lu, lv, le

    def fe_preprocess(self, tu_copy, tv_copy, lu_copy, lv_copy, le_copy):
        """GraphTuneの出力を確率密度、またはArgmaxによるonehot表現へ変換する。
        また、Argmaxによって確定されたグラフのobj, conditionに対応するグラフ特徴量を取得する。
        さらに、ArgmaxによってわかったEOS以降のシーケンスは0にする。

        Args:
            tu_copy (torch.Tensor): timestamp tu
            tv_copy (torch.Tensor): timestamp tv
            lu_copy (torch.Tensor): node label lu
            lv_copy (torch.Tensor): node label lv
            le_copy (torch.Tensor): edge label le

        Returns:
            (list): [dfs code,
                    Argmaxによって確定されたグラフのconditionに対応する特徴量,
                    Argmaxによって確定されたグラフのobject]
        """
        # 確率密度に変換
        target_list = [tu_copy, tv_copy, lu_copy, lv_copy, le_copy]
        if not self.flag_softmax:
            for l_i, target in enumerate(target_list):
                target_list[l_i] = self.softmax(target)

        # Argmaxをとる
        indicies_for_onehot = []
        if self.fe_preprocess_onehot:
            # Get onehot encodings
            for l_i, target in enumerate(target_list):
                indicies_for_onehot.append(torch.argmax(target, dim=2))
                target_list[l_i] = F.one_hot(indicies_for_onehot[l_i], num_classes=target.shape[2])
        else:
            # Get probability distributions
            for l_i, target in enumerate(target_list):
                indicies_for_onehot.append(torch.argmax(target, dim=2))

        # tu, tv, lu, lv, leを連結させる
        indicies_for_onehot = [code.unsqueeze(dim=2) for code in indicies_for_onehot]
        dfs_code = torch.cat(indicies_for_onehot, dim=2)

        # グラフオブジェクトに変換し、グラフ特徴量を計算する
        if self.fe_preprocess_convert_graph_obj:
            # DFS codeをnetworkxのグラフobjへ変換する
            generated_graph = []
            for code in dfs_code:
                graph = graph_utils.dfs_code_to_graph_obj(
                    code.cpu().detach().numpy(),
                    [self.time_size, self.time_size, self.node_size, self.node_size, self.edge_size],
                    edge_num=tu_copy.shape[1])
                generated_graph.append(graph)
            # conditonに対応する特徴量を計算する
            st = graph_statistic.GraphStatistic()
            generated_labels = torch.Tensor()
            for graph in generated_graph:
                ## paramsはリスト型で渡されるのでindex[0]をつける
                calculated_params = st.calc_graph_traits2csv([graph], self.condition_params)[0]
                tmp_label = []
                for param in calculated_params.values():
                    if param is None:
                        ## 特徴量が例外により計算できなかった場合、値を0とする
                        tmp_label.append(0)
                    else:
                        tmp_label.append(round(param, self.condition_round))
                if len(tmp_label) == 1:
                    tmp_label = torch.tensor(tmp_label).float().unsqueeze(dim=0)
                else:
                    tmp_label = torch.tensor(np.array([np.prod(tmp_label)])).float().unsqueeze(dim=0)
                generated_labels = torch.cat((generated_labels, tmp_label), dim=0)
            generated_labels = generated_labels.squeeze(dim=1)
        else:
            generated_labels = None
            generated_graph = None

        # EOS以降の値を0にする
        for i in range(0, dfs_code.shape[0], 1):
            # EOSのindexを探す
            ## timestamp tuのEOSのindexを探す
            eos_indicies = (dfs_code[i].T[0]==self.time_size-1).nonzero().squeeze(dim=1)
            ## timestamp tvのEOSのindexを探す
            eos_indicies = torch.cat((eos_indicies, (dfs_code[i].T[1]==self.time_size-1).nonzero().squeeze(dim=1)), dim=0)
            ## node label luのEOSのindexを探す
            eos_indicies = torch.cat((eos_indicies, (dfs_code[i].T[2]==self.node_size-1).nonzero().squeeze(dim=1)), dim=0)
            ## node label lvのEOSのindexを探す
            eos_indicies = torch.cat((eos_indicies, (dfs_code[i].T[3]==self.node_size-1).nonzero().squeeze(dim=1)), dim=0)
            ## edge label leのEOSのindexを探す
            eos_indicies = torch.cat((eos_indicies, (dfs_code[i].T[4]==self.edge_size-1).nonzero().squeeze(dim=1)), dim=0)

            # EOSが見つからなければ、処理をskip
            if torch.equal(eos_indicies, torch.empty(0, device=self.device)):
                continue

            # EOSのindiciesの内、最小のindex(EOS閾値)を得る
            eos_th = min(eos_indicies) + 1

            # EOS閾値が最大シーケンス長以上であれば、処理をskip
            if eos_th >= target_list[0].shape[1] - 1:
                continue

            # EOS閾値を使って、EOS以降のシーケンスデータを0にする
            for l_i, _ in enumerate(target_list):
                target_list[l_i][i] = torch.cat(
                    (target_list[l_i][i][:eos_th], torch.zeros(target_list[l_i].shape[1]-eos_th, target_list[l_i].shape[2], device=self.device)),
                    dim=0
                )

        # Return [DFS code, conditionで指定したグラフ特徴量, DFS codeから変換されたグラフobj]
        return torch.cat([target_list[0], target_list[1], target_list[2], target_list[3], target_list[4]], dim=2), generated_labels, generated_graph



if __name__ == "__main__":
    print("cvae_with_fe.py")
    args = {
        'softmax': True,
    }
    import config
    params = config.Parameters(args)
    model = CVAEwithFE(dfs_size=173, time_size=51, node_size=34, edge_size=2, condition_size=1, params=params, device="cuda")
    print(model)