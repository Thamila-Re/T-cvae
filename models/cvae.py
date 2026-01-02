"""
Conditional Variational AutoEncoder(CVAE) を定義するモジュール.
"""

import torch
from torch import nn
import sys
import os
import torch.nn.functional as F
import numpy as np
import random
import networkx as nx
import argparse



sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import try_gpu, sample_dist, convert2onehot, sample_topk_topp
from graph_process import graph_utils
from config import common_args, Parameters
import utils
from utils import dump_params, setup_params


class CVAE(nn.Module):
    """Conditional VAE class

    input_data => CVAE(Encoder, Decoder) => output_data
    """
    def __init__(self, dfs_size, time_size, node_size, edge_size, condition_size, params, device):
        super(CVAE, self).__init__()
        emb_size = params.model_params["emb_size"]
        en_hidden_size = params.model_params["en_hidden_size"]
        de_hidden_size = params.model_params["de_hidden_size"]
        self.rep_size = params.model_params["rep_size"]
        self.alpha = params.model_params["alpha"]
        self.beta = params.model_params["beta"]
        self.word_drop = params.model_params["word_drop"]
        self.time_size = time_size
        self.node_size = node_size
        self.edge_size = edge_size
        self.device = device
        self.condition_size = condition_size

        if params.model_params["encoder_condition"]:
            self.encoder = Encoder(dfs_size, emb_size, en_hidden_size, self.rep_size, self.device, condition_size=self.condition_size, use_condition=params.model_params["encoder_condition"])
        else:
            # conditionが入力にない場合、input_sizeをconditionの分小さくする
            self.encoder = Encoder(dfs_size-self.condition_size, emb_size, en_hidden_size, self.rep_size, self.device, condition_size=self.condition_size, use_condition=params.model_params["encoder_condition"])

        if params.model_params["decoder_h_c_condition"] is True and params.model_params["decoder_sequence_condition"] is False:
            self.decoder = Decoder(self.rep_size, dfs_size-condition_size, emb_size, de_hidden_size, time_size, node_size, edge_size, condition_size, params, \
                self.device, h_c_condition=params.model_params["decoder_h_c_condition"], seq_condition=params.model_params["decoder_sequence_condition"])
        else:
            self.decoder = Decoder(self.rep_size, dfs_size, emb_size, de_hidden_size, time_size, node_size, edge_size, condition_size, params, \
                self.device, h_c_condition=params.model_params["decoder_h_c_condition"], seq_condition=params.model_params["decoder_sequence_condition"])

    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.transformation(mu, sigma, self.device)
        tu, tv, lu, lv, le = self.decoder(z, x)
        return mu, sigma, tu, tv, lu, lv, le

    @staticmethod
    def transformation(mu, sigma, device):
        """Reparametrization trick

        mu, sigma, 正規分布から取得したノイズから潜在変数zを計算する.

        Args:
            mu (_type_): _description_
            sigma (_type_): _description_
            device (_type_): _description_

        Returns:
            _type_: _description_
        """
        return mu + torch.exp(0.5 * sigma) * torch.randn(sigma.shape).to(device)

    def loss(self, encoder_loss, decoder_loss):
        """Loss function

        Args:
            encoder_loss (_type_): Encoder modelのloss
            decoder_loss (_type_): Decoder modelのloss

        Returns:
            (): CVAEのloss
        """
        cvae_loss = self.beta * encoder_loss + self.alpha * decoder_loss
        return cvae_loss

    def generate(self, data_num, conditional_label, max_size, z=None, is_output_sampling=True):
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
        # 従来の実装
        if z is None:
            z = self.noise_generator(self.rep_size, data_num).unsqueeze(1)
            z = z.to(self.device)
        tu, tv, lu, lv, le =\
            self.decoder.generate(z, conditional_label, max_size, is_output_sampling)
        
        # 確実にdata_num個のグラフを生成する実装
        # if z is None:
        #     z = self.noise_generator(self.rep_size, 1).unsqueeze(1)
        #     z = z.to(self.device)
        # tu = torch.LongTensor()
        # tv = torch.LongTensor()
        # lu = torch.LongTensor()
        # lv = torch.LongTensor()
        # le = torch.LongTensor()
        # is_sufficient_size = lambda graph: True if graph.number_of_nodes() > 0 else False
        # generated_graph_num = 0
        # while generated_graph_num < data_num:
        #     tu_one, tv_one, lu_one, lv_one, le_one =\
        #         self.decoder.generate(z, conditional_label, max_size, is_output_sampling)
        #     [tu_one, tv_one, lu_one, lv_one, le_one] = [code.unsqueeze(2) for code in [tu_one, tv_one, lu_one, lv_one, le_one]]
        #     dfs_code = torch.cat([tu_one, tv_one, lu_one, lv_one, le_one], dim=2)
        #     graph = graph_utils.dfs_code_to_graph_obj(
        #         dfs_code[0].cpu().detach().numpy(),
        #         [self.time_size, self.time_size, self.node_size, self.node_size, self.edge_size],
        #         edge_num=max_size)
        #     if nx.is_connected(graph) and is_sufficient_size(graph):
        #         tu = torch.cat((tu, tu_one), dim=0)
        #         tv = torch.cat((tv, tv_one), dim=0)
        #         lu = torch.cat((lu, lu_one), dim=0)
        #         lv = torch.cat((lv, lv_one), dim=0)
        #         le = torch.cat((le, le_one), dim=0)
        #         generated_graph_num += 1
        # tu = tu.squeeze(dim=2)
        # tv = tv.squeeze(dim=2)
        # lu = lu.squeeze(dim=2)
        # lv = lv.squeeze(dim=2)
        # le = le.squeeze(dim=2)    
        
        return tu, tv, lu, lv, le

    def noise_generator(self, rep_size, batch_num):
        """Generate noise

        Args:
            rep_size  ():
            batch_num ():

        Returns:
            ():
        """
        return torch.randn(batch_num, rep_size)

    def generate_with_history(self, data_num: int, conditional_label: torch.Tensor, 
                              max_size: int, z: torch.Tensor=None):
        """Generate graph samples

        Args:
            data_num                   (int): 生成サンプル数
            conditional_label (torch.Tensor): 条件として与えるラベル情報
            max_size                   (int): 最大エッジ数
            z                 (torch.Tensor): 潜在空間からサンプリングされたデータ
            is_output_sampling        (bool): Trueなら返り値を予測dfsコードからargmaxしたものに. Falseなら予測分布を返す
        Returns:
            (torch.Tensor): 生成されたサンプルの5-tuplesの各要素のデータとサンプリングで何番目の確率が選ばれたかの記録
        """
        # 従来の実装
        if z is None:
            z = self.noise_generator(self.rep_size, data_num).unsqueeze(1)
            z = z.to(self.device)
        tu, tv, lu, lv, le, sampling_record =\
            self.decoder.generate_with_history(z, conditional_label, max_size)
        
        return (tu, tv, lu, lv, le), sampling_record


class Encoder(nn.Module):
    """
    Encoderクラス.

    線形層1(input_size, emb_size) => LSTM(emb_size, hidden_size) => 線形層2(hidden_size, rep_size)
    """

    def __init__(self, input_size, emb_size, hidden_size, rep_size, device, condition_size,num_layer=2, use_condition=True):
        """Encoderのハイパーパラメータを設定する.

        Args:
            input_size  (int) : 入力データのサイズ
            emb_size    (int) : 埋め込み層への入力サイズ
            hidden_size (int) : LSTMの隠れ層のサイズ
            rep_size    (int) : LSTMの後に配置される線形層の出力サイズ
            device      ()    : デバイス情報
            num_layer (int, optional): LSTMの層数. Defaults to 2.
        """
        super(Encoder, self).__init__()
        self.emb = nn.Linear(input_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers=num_layer, batch_first=True)
        self.mu = nn.Linear(hidden_size, rep_size)
        self.sigma = nn.Linear(hidden_size, rep_size)
        self.device = device
        self.num_layer = num_layer
        self.hidden_size = hidden_size

        self.use_condition = use_condition
        self.condition_size = condition_size

    def forward(self, x):
        """順伝搬

        Args:
            x (_type_): DFSコードのミニバッチ

        Returns:
            () : 分布の平均値に該当する値
            () : 分布の標準偏差に該当する値
        """
        if self.use_condition is False:
            # 入力xにconditionが付与されているので取り除く
            x = x[:,:,:-1*self.condition_size]
        x = self.emb(x)
        # x = torch.cat((x,label),dim=2)
        x, (h, c) = self.lstm(x)
        x = x[:, -1, :].unsqueeze(1)
        return self.mu(x), self.sigma(x)

    def loss(self, mu, sigma):
        """正規分布とのKL divergenceを計算する関数

        Args:
            mu    () : 分布の平均値に該当する値
            sigma () : 分布の分散に該当する値

        Returns:
            () : KL divergence
        """
        delta = 1e-7
        return -0.5 * torch.sum(1 + sigma - mu ** 2 - torch.exp(sigma + delta))


class Decoder(nn.Module):
    """
    Decoder クラス.

    () => () => () => () => ()
    """

    def __init__(self, rep_size, input_size, emb_size, hidden_size, time_size, node_label_size, edge_label_size,
                 condition_size, params, device, num_layer=3, h_c_condition=True, seq_condition=True):
        """Decoderのハイパーパラメータを設定する.

        Args:
            rep_size (_type_): _description_
            input_size (_type_): _description_
            emb_size (_type_): _description_
            hidden_size (_type_): _description_
            time_size (_type_): _description_
            node_label_size (_type_): _description_
            edge_label_size (_type_): _description_
            condition_size (_type_): _description_
            params (_type_): configで設定されたglobal変数のset
            device (_type_): _description_
            num_layer (int, optional): _description_. Defaults to 3.
            use_condition (bool, optional): conditionを使用するかどうか 通常は使用.
        """
        super(Decoder, self).__init__()
        self.h_c_condition = h_c_condition
        self.seq_condition = seq_condition
        self.use_decoder_femb = params.use_decoder_femb ## f_embを使うかどうかを制御するフラグ (default:True)
        '''初期状態h0,c0の生成モード
            - 0 : 特徴量CでFillする (default)
            - 1 : 0でFillする
            - 2 : 潜在変数zを線形層f_h0c0に通す latent_size -> hidden_size
            - 3 : 潜在変数z,特徴量Cを線形層f_h0c0に通す latent_size+condition_size -> hidden_size
        '''
        self.h0c0_mode = params.h0c0_mode 
        self.cat_rep_to_sos_and_input = params.cat_rep_to_sos_and_input # repをSOSと入力に連結するかどうか
        '''初期状態を生成する線形層を定義する'''
        if self.h0c0_mode == 0 or self.h0c0_mode == 1:
            pass # 線形層は定義しない
        elif self.h0c0_mode == 2:
            self.f_h0c0 = nn.Linear(rep_size, hidden_size) # 潜在変数zを線形層f_h0c0に通す latent_size -> hidden_size
        elif self.h0c0_mode == 3:
            self.f_h0c0 = nn.Linear(rep_size+condition_size, hidden_size) # 潜在変数z,特徴量Cを線形層f_h0c0に通す latent_size+condition_size -> hidden_size
        else:
            raise ValueError('h0c0_mode is invalid')
        '''SOSの生成モード
            - 0 : 潜在変数z, 特徴量Cを線形層f_rep(f_dsos)に通す (default)
            - 1 : 0埋めに特徴量Cを連結する
            - 2 : 特徴量Cを線形層f_repに通す
            - 3 : 初期状態h_0からSOSを計算する
        '''
        self.sos_mode = params.sos_mode
        if self.sos_mode == 0:
            self.f_rep = nn.Linear(rep_size+condition_size, input_size) # 潜在変数z, 特徴量Cを線形層f_repに通す latent_size+condition_size -> input_size
        elif self.sos_mode == 2:
            self.f_rep = nn.Linear(condition_size, input_size) # 特徴量Cを線形層f_repに通す condition_size -> input_size
        elif self.sos_mode == 1 or self.sos_mode == 3:
            pass # 線形層は定義しない
        else:
            raise ValueError('sos_mode is invalid')
        self.sampling_generation = params.sampling_generation
        self.condition_size = condition_size
        self.num_layer = num_layer
        self.input_size = input_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.emb = nn.Linear(input_size, emb_size) # f_emb : input_size -> emb_size
        if self.use_decoder_femb:
            lstm_input_size = emb_size
        else:
            lstm_input_size = input_size
        if self.cat_rep_to_sos_and_input == True:
            lstm_input_size += rep_size + condition_size
        self.lstm = nn.LSTM(lstm_input_size, hidden_size, num_layers=self.num_layer,
                        batch_first=True)
        self.tuple_name = params.tuple_name # get setted tuple name
        self.f_tu = nn.Linear(hidden_size, time_size)
        self.f_tv = nn.Linear(hidden_size, time_size)
        self.f_lu = nn.Linear(hidden_size, node_label_size)
        self.f_lv = nn.Linear(hidden_size, node_label_size)
        self.f_le = nn.Linear(hidden_size, edge_label_size)
        # 以下は、直接LLを定義してもいい
        self.f_tuple_dict = {self.tuple_name[0]: self.f_tu,
                             self.tuple_name[1]: self.f_tv,
                             self.tuple_name[2]: self.f_lu,
                             self.tuple_name[3]: self.f_lv,
                             self.tuple_name[4]: self.f_le}
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(0.5)
        
        self.flag_softmax = params.args["softmax"]

        self.f_h = nn.Linear(hidden_size, hidden_size)
        self.f_c = nn.Linear(hidden_size, hidden_size)

        self.time_size = time_size
        self.node_label_size = node_label_size
        self.edge_label_size = edge_label_size
        self.tuple_size_dict = {self.tuple_name[0]: self.time_size,
                                self.tuple_name[1]: self.time_size,
                                self.tuple_name[2]: self.node_label_size,
                                self.tuple_name[3]: self.node_label_size,
                                self.tuple_name[4]: self.edge_label_size}

        self.ignore_label = params.ignore_label
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_label, reduction="sum")
        self.device = device

        self.softmax_temperature = params.softmax_temperature
        self.sampling_mode = params.sampling_mode
        self.top_k = params.top_k
        self.top_p = params.top_p

    def __calc_h0c0(self, rep, conditional):
        ''' 初期状態h0,c0を生成する関数
            Description:
                初期状態h0,c0の生成モード
                - 0 : 特徴量CでFillする (default)
                - 1 : 0でFillする
                - 2 : 潜在変数zを線形層f_h0c0に通す
                - 3 : 潜在変数z,特徴量Cを線形層f_h0c0に通す
            Args:
                rep: 潜在変数z
                conditional: 特徴量C
            Returns:
                h_0: 初期状態h0
                c_0: 初期状態c0
        '''
        if self.h0c0_mode == 0: # default
            h_0 = conditional.permute(1,0,2).repeat(self.num_layer,1,self.hidden_size)
            c_0 = conditional.permute(1,0,2).repeat(self.num_layer,1,self.hidden_size)
        elif self.h0c0_mode == 1:
            h_0 = torch.Tensor(self.num_layer, conditional.shape[0], self.hidden_size).fill_(0).to(self.device)
            c_0 = torch.Tensor(self.num_layer, conditional.shape[0], self.hidden_size).fill_(0).to(self.device)
        elif self.h0c0_mode == 2:
            # 潜在変数zを線形層f_h0c0に通す
            h_0 = self.f_h0c0(rep) # latent_size -> hidden_size
            h_0 = torch.permute(h_0, (1,0,2)) # (batch, 1, hidden_size) -> (1, batch, hidden_size)
            h_0 = h_0.repeat(self.num_layer,1,1) # (1, batch, hidden_size) -> (num_layer, batch, hidden_size)

            c_0 = self.f_h0c0(rep) # latent_size -> hidden_size
            c_0 = torch.permute(c_0, (1,0,2)) # (batch, 1, hidden_size) -> (1, batch, hidden_size)
            c_0 = c_0.repeat(self.num_layer,1,1) # (1, batch, hidden_size) -> (num_layer, batch, hidden_size)
        elif self.h0c0_mode == 3:
            # 潜在変数z,特徴量Cを線形層f_h0c0に通す
            h_0 = self.f_h0c0(torch.cat((rep, conditional), dim=2)) # latent_size + condition_size -> hidden_size
            h_0 = torch.permute(h_0, (1,0,2)) # (batch, 1, hidden_size) -> (1, batch, hidden_size)
            h_0 = h_0.repeat(self.num_layer,1,1) # (1, batch, hidden_size) -> (num_layer, batch, hidden_size)

            c_0 = self.f_h0c0(torch.cat((rep, conditional), dim=2)) # latent_size + condition_size -> hidden_size
            c_0 = torch.permute(c_0, (1,0,2)) # (batch, 1, hidden_size) -> (1, batch, hidden_size)
            c_0 = c_0.repeat(self.num_layer,1,1) # (1, batch, hidden_size) -> (num_layer, batch, hidden_size)
        else:
            raise ValueError("h0c0_mode is invalid.")
        return h_0, c_0

    def __calc_sos(self, rep, conditional, initial_hidden_state:torch.tensor=None,
                   mode:str = None):
        ''' SOSの生成するための関数
            Description:
                SOSの生成モード
                - 0 : 潜在変数z, 特徴量Cを線形層f_sos(f_rep)に通す
                - 1 : 0埋めに特長量Cを連結する
                - 2 : 特徴量Cを線形層f_sos(f_rep)に通す
                - 3 : 初期状態h0,c0から生成する。(大規模な改修が必要なため、未実装。井脇)
            Args:
                rep: 潜在変数z
                conditional: 特徴量C
            Returns:
                sos: SOS
        '''
        sos = None
        if self.sos_mode == 0:
            sos = self.f_rep(torch.cat((rep, conditional), dim=2))
        elif self.sos_mode == 1:
            sos = torch.cat([torch.zeros(conditional.shape[0], 1, self.input_size-1).to(self.device), conditional], dim=2)
        elif self.sos_mode == 2:
            sos = self.f_rep(conditional)
        elif self.sos_mode == 3:
            if initial_hidden_state is None:
                raise ValueError("initial_hidden_state is None.")
            elif mode is None:
                raise ValueError("mode is None.")
            x = initial_hidden_state.permute(1,0,2)[:,0,:].unsqueeze(1)
            next_tuple_dict = self.__calc_next_tuple(x, 
                                            all_tuple_dist_dict = None,
                                            all_tuple_value_dict = None,
                                            all_tuple_rank_dict = None,
                                            mode=mode)
            tu, tv, lu, lv, le = next_tuple_dict['each_tuple_dist_dict'].values() # 辞書から値の取り出し
            sos = torch.cat((tu, tv, lu, lv, le), dim=2).to(self.device)
            sos = torch.cat((sos, conditional), dim=2)
        else:
            raise ValueError("sos_mode is invalid.")
        if sos is None:
            raise ValueError("sos is None.")
        return sos
    
    def __calc_next_tuple(self, x: torch.Tensor, 
                                all_tuple_dist_dict: dict=None,
                                all_tuple_value_dict: dict=None,
                                all_tuple_rank_dict: dict=None,
                                mode: str=None)->dict:
        ''' Generate next tuple values
            Args:
                x: lstm output or initial hidden state h0
                all_tuple_dist_dict: dict=None : 
                all_tuple_value_dict: dict=None : 
                all_tuple_rank_dict: dict=None :
            Variables(Dictionalies):
                all_tuple_dist_dict: dict : 全ての5-tupleの値をサンプリングする分布を格納する辞書。evalのみ用いる。
                all_tuple_value_dict: dict : 全ての5-tupleの値を格納する辞書。evalのみ用いる。
                all_tuple_rank_dict: dict : evalのみ用いる。
                each_tuple_dist_dict: dict : 次の5-tupleの値をサンプリングする分布を格納する辞書。
                each_tuple_value_dict: dict : 次の5-tupleの値を格納する辞書。
                each_tuple_rank_dict: dict : 
            Returns:
                next_tuple_dict: dict : 次の5-tupleの値に関する全ての辞書を格納する辞書。
        '''
        if mode is None:
            raise ValueError("mode is None.")
        elif mode == "train":
            softmax_temperature = 1.0
        elif mode == "eval":
            softmax_temperature = self.softmax_temperature

        each_tuple_dist_dict = {} # 次の5-tupleの値をサンプリングする分布を格納する辞書
        each_tuple_value_dict = {} # 次の5-tuple形式のdfsコードを格納する辞書
        each_tuple_rank_dict = {} 

        # Initialize all_tuple_dist_dict
        if all_tuple_dist_dict is None:
            all_tuple_dist_dict = {} # 全ての5-tupleの値をサンプリングする分布を格納する辞書
            for name in self.tuple_name:
                all_tuple_dist_dict[name] = torch.Tensor().to(self.device)
        
        # Initialize all_tuple_value_dict
        if all_tuple_value_dict is None:
            all_tuple_value_dict = {} # 全ての5-tuple形式のdfsコードを格納する辞書
            for name in self.tuple_name:
                all_tuple_value_dict[name] = torch.LongTensor().to(self.device)

        # Initialize all_tuple_rank_dict
        if all_tuple_rank_dict is None:
            all_tuple_rank_dict = {} 
            for name in self.tuple_name:
                all_tuple_rank_dict[name] = torch.LongTensor().to(self.device)
                
        for name in self.tuple_name:
            each_tuple_dist_dict[name]=self.softmax(self.f_tuple_dict[name](x)/softmax_temperature)

        if mode == "train":
            # train時は以降の計算に必要は引数を与えられないので、ここで計算を終了する。
            next_tuple_dict = {
                'all_tuple_dist_dict': all_tuple_dist_dict,
                'all_tuple_value_dict': all_tuple_value_dict,
                'all_tuple_rank_dict': all_tuple_rank_dict,
                'each_tuple_dist_dict': each_tuple_dist_dict,
                'each_tuple_value_dict': each_tuple_value_dict,
                'each_tuple_rank_dict': each_tuple_rank_dict
            }
            return next_tuple_dict

        ## 計算した確率分布をall_tuple_dist_dictに結合する
        for name, dist in all_tuple_dist_dict.items():
            all_tuple_dist_dict[name] = torch.cat((dist, each_tuple_dist_dict[name]), dim=1)

        # samping 5-tuple values from calculated distribution
        ## 5-tupleの値をサンプリングする
        if self.sampling_mode!='none':
            for name in self.tuple_name:
                each_tuple_value_dict[name], each_tuple_rank_dict[name] = sample_topk_topp(
                                                                            each_tuple_dist_dict[name],
                                                                            k=self.top_k,p=self.top_p, 
                                                                            mode=self.sampling_mode)
        elif self.sampling_generation:
            for name in self.tuple_name:
                each_tuple_value_dict[name], each_tuple_rank_dict[name] = sample_dist(each_tuple_dist_dict[name])
        else:
            for name in self.tuple_name:
                each_tuple_value_dict[name] = torch.argmax(each_tuple_dist_dict[name], dim=2)

        ## 計算した5-tupleの値をall_tuple_value_dictに結合する
        for name, tuple_value in all_tuple_value_dict.items():
            all_tuple_value_dict[name] = torch.cat((tuple_value, each_tuple_value_dict[name]), dim=1)

        ## 計算した5-tupleのrankをall_tuple_rank_dictに結合する
        for name, tuple_rank in all_tuple_rank_dict.items():
            all_tuple_rank_dict[name] = torch.cat((tuple_rank, each_tuple_rank_dict[name]), dim=1)

        ## 5-tupleの値をone-hotベクトルに変換
        for name, tuple_value in each_tuple_value_dict.items():
            each_tuple_value_dict[name] = F.one_hot(tuple_value, self.tuple_size_dict[name]).squeeze()

        next_tuple_dict = {
            'all_tuple_dist_dict': all_tuple_dist_dict,
            'all_tuple_value_dict': all_tuple_value_dict,
            'all_tuple_rank_dict': all_tuple_rank_dict,
            'each_tuple_dist_dict': each_tuple_dist_dict,
            'each_tuple_value_dict': each_tuple_value_dict,
            'each_tuple_rank_dict': each_tuple_rank_dict
            }
        
        return next_tuple_dict
    
    def forward(self, rep: torch.Tensor, x: torch.Tensor, word_drop=0):
        """
            学習時のforward
            Args:
                rep: torch.Tensor : encoderの出力(潜在変数)
                x: torch.Tensor : dfs code
            Returns:
                tu: source time
                tv: sink time
                lu: source node label
                lv: sink node label
                le: edge label
        """
        conditional_label = x[:, 0, -1 * self.condition_size:].unsqueeze(1)

        if self.seq_condition:
            extended_rep = torch.cat([rep, conditional_label], dim=2)
        else:
            extended_rep = rep
        
        if self.seq_condition is False:
            # 入力xにconditionが付与されているので取り除く
            x = x[:,:,:-1*self.condition_size]

        # generate h0,c0 below
        h_0, c_0 = self.__calc_h0c0(rep, conditional_label)

        # generate SOS below
        sos = self.__calc_sos(rep, conditional_label, h_0, mode='train') # X -> input_size
        # sos = self.dropout(sos)
        ''''以下でdfs_codeにsosを連結
            サイズがオーバーするので最後の一つを削除する'''
        x = torch.cat((sos, x), dim=1)[:, :-1, :] 

        # word drop
        for batch in range(x.shape[0]):
            args = random.choices([i for i in range(x.shape[1])], k=int(x.shape[1] * word_drop))
            zero = try_gpu(self.device, torch.zeros([1, 1, x.shape[2] - self.condition_size]))
            x[batch, args, :-1 * self.condition_size] = zero
        if self.use_decoder_femb:
            x = self.emb(x) # f_emb : input_size -> emb_size
            
        extended_rep = extended_rep.repeat(1,x.shape[1],1)
        
        if self.cat_rep_to_sos_and_input:
            x = torch.cat((x, extended_rep), dim=2) # 入力データにrep(潜在変数+特徴量)を連結

        x, (h, c) = self.lstm(x, (h_0, c_0)) # xのサイズを間違えないように注意する。井脇
        x = self.dropout(x)

        next_tuple_dict = self.__calc_next_tuple(x, 
                                                 all_tuple_dist_dict = None,
                                                 all_tuple_value_dict = None,
                                                 all_tuple_rank_dict = None,
                                                 mode='train')

        tu, tv, lu, lv, le = next_tuple_dict['each_tuple_dist_dict'].values() # 辞書から値の取り出し
        
        # Rrefactoring : 部分的に辞書化する。本当は、辞書をreturnしたい。井脇
        return tu, tv, lu, lv, le

    def generate(self, rep: torch.Tensor, conditional_label: torch.Tensor, 
                 max_size: int=100, is_output_sampling: bool=True):
        """
            生成時のforward. 生成したdfsコードを用いて、新たなコードを生成していく
            Args:
                rep: torch.Tensor : encoderの出力。潜在変数
                conditional_label: torch.Tensor : 条件付き生成のための特徴量のラベル
                max_size: int : 生成を続ける最大サイズ(生成を続けるエッジの最大数)
                is_output_sampling: bool : Trueなら返り値を予測dfsコードからサンプリングしたものを. Falseなら予測分布を返す
            Returns:
        """

        conditional_label = conditional_label.repeat(rep.shape[0], 1, 1)

        if self.seq_condition:
            extended_rep = torch.cat([rep, conditional_label], dim=2) # 潜在変数zにconditionを連結
        else:
            extended_rep = rep
            
        # generate h0,c0 below
        h_0, c_0 = self.__calc_h0c0(rep, conditional_label)

        # generate SOS below
        sos = self.__calc_sos(rep, conditional_label, h_0, mode='eval') # X -> input_size

        if self.use_decoder_femb: # Embedding Layer
            sos = self.emb(sos) # f_emb : input_size -> emb_size

        if self.cat_rep_to_sos_and_input: # concat rep to sos
            sos = torch.cat((sos, extended_rep), dim=2) # 入力データにrep(潜在変数+特徴量)を連結

        # Initialize dictionlies
        all_tuple_dist_dict = {} # 全ての5-tupleの値をサンプリングする分布を格納する辞書
        for name in self.tuple_name:
            all_tuple_dist_dict[name] = torch.Tensor().to(self.device)

        all_tuple_value_dict = {} # 全ての5-tuple形式のdfsコードを格納する辞書
        for name in self.tuple_name:
            all_tuple_value_dict[name] = torch.LongTensor().to(self.device)

        all_tuple_rank_dict = {} 
        for name in self.tuple_name:
            all_tuple_rank_dict[name] = torch.LongTensor().to(self.device)

        for i in range(max_size):
            if i == 0:
                x, (h, c) = self.lstm(sos, (h_0, c_0))
            else:
                if self.use_decoder_femb:
                    x = self.emb(x) # f_emb : input_size -> emb_size
                if self.cat_rep_to_sos_and_input:
                    x = torch.cat((x, extended_rep), dim=2) # 入力データにrep(潜在変数+特徴量)を連結
                x, (h, c) = self.lstm(x, (h, c))

            next_tuple_dict = self.__calc_next_tuple(x, 
                                        all_tuple_dist_dict = all_tuple_dist_dict,
                                        all_tuple_value_dict = all_tuple_value_dict,
                                        all_tuple_rank_dict = all_tuple_rank_dict,
                                        mode='eval')
            
            tu, tv, lu, lv, le = next_tuple_dict['each_tuple_value_dict'].values()
            x = torch.cat((tu, tv, lu, lv, le), dim=1).unsqueeze(1).to(self.device)

            if self.seq_condition:
                x = torch.cat((x, conditional_label), dim=2)

        if is_output_sampling:
            tus, tvs, lus, lvs, les = next_tuple_dict['all_tuple_value_dict'].values()
            return tus, tvs, lus, lvs, les
        else:
            tus_dist, tvs_dist, lus_dist, lvs_dist, les_dist \
                = next_tuple_dict['all_tuple_dist_dict'].values()
            return tus_dist, tvs_dist, lus_dist, lvs_dist, les_dist

    def loss(self, results, targets):
        """Cross Entropyを計算する関数

        Args:
            results   (dict): Decoderから出力されたDFSコードの分布
            targets   (dict): labelデータ

        Returns:
            (dict): 5-tuplesの各要素のlossを持つdict
            (): 5-tuplesの各要素のlossのsum
        """
        total_loss = 0
        loss_dict = {}
        for i, (key, pred) in enumerate(results.items()):
            loss_dict[key] = self.criterion(pred.transpose(2, 1), targets[key])
            total_loss += loss_dict[key]
        return loss_dict.copy(), total_loss

    def accuracy(self, results, targets):
        """分類精度を計算する関数

        確率分布からargmaxを取ることでサンプリングする.

        Args:
            results (_type_): _description_
            targets (_type_): _description_

        Returns:
            (dict): 5-tuplesの各要素の分類精度
        """
        acc_dict = {}
        for i, (key, pred) in enumerate(results.items()):
            pred = torch.argmax(pred, dim=2)  # onehot => label
            pred = pred.view(-1)
            targets[key] = targets[key].view(-1)
            # 分類精度を計算
            score = torch.zeros(pred.shape[0])
            score[pred == targets[key]] = 1
            data_len = pred.shape[0]
            if not self.ignore_label is None:
                targets[key] = targets[key].cpu()
                ignore_args = np.where(targets[key] == self.ignore_label)[0]
                data_len -= len(ignore_args)
                score[ignore_args] = 0
            score = torch.sum(score) / data_len
            acc_dict[key] = score
        return acc_dict.copy()
    
    def generate_with_history(self, rep: torch.Tensor, conditional_label:torch.Tensor, 
                              max_size: int=100):
        """
        サンプリング記録を残して生成する時のforward. 生成したdfsコードを用いて、新たなコードを生成していく
        Args:
            rep: encoderの出力(潜在変数)
            max_size: 生成を続ける最大サイズ(生成を続けるエッジの最大数)
            is_output_sampling: Trueなら返り値を予測dfsコードからサンプリングしたものを. Falseなら予測分布を返す
        Returns:
        """

        conditional_label = conditional_label.repeat(rep.shape[0], 1, 1)

        if self.seq_condition:
            extended_rep = torch.cat([rep, conditional_label], dim=2) # 潜在変数zにconditionを連結
        else:
            extended_rep = rep
    
        # generate h0,c0 below
        h_0, c_0 = self.__calc_h0c0(rep, conditional_label)

        # generate SOS below
        sos = self.__calc_sos(rep, conditional_label, h_0, mode='eval') # X -> input_size

        if self.use_decoder_femb: # Embedding Layer
            sos = self.emb(sos) # f_emb : input_size -> emb_size

        if self.cat_rep_to_sos_and_input: # concat rep to sos
            sos = torch.cat((sos, extended_rep), dim=2) # 入力データにrep(潜在変数+特徴量)を連結

        # Initialize dictionlies
        all_tuple_dist_dict = {} # 全ての5-tupleの値をサンプリングする分布を格納する辞書
        for name in self.tuple_name:
            all_tuple_dist_dict[name] = torch.Tensor().to(self.device)

        all_tuple_value_dict = {} # 全ての5-tuple形式のdfsコードを格納する辞書
        for name in self.tuple_name:
            all_tuple_value_dict[name] = torch.LongTensor().to(self.device)

        all_tuple_rank_dict = {} 
        for name in self.tuple_name:
            all_tuple_rank_dict[name] = torch.LongTensor().to(self.device)

        for i in range(max_size):
            if i == 0:
                x, (h, c) = self.lstm(sos, (h_0, c_0))
            else:
                if self.use_decoder_femb:
                    x = self.emb(x) # f_emb : input_size -> emb_size
                if self.cat_rep_to_sos_and_input:
                    x = torch.cat((x, extended_rep), dim=2)
                x, (h, c) = self.lstm(x, (h, c))

            next_tuple_dict = self.__calc_next_tuple(x, 
                                        all_tuple_dist_dict = all_tuple_dist_dict,
                                        all_tuple_value_dict = all_tuple_value_dict,
                                        all_tuple_rank_dict = all_tuple_rank_dict,
                                        mode='eval')
            
            tu, tv, lu, lv, le = next_tuple_dict['each_tuple_value_dict'].values()
            x = torch.cat((tu, tv, lu, lv, le), dim=1).unsqueeze(1).to(self.device)

            if self.seq_condition:
                x = torch.cat((x, conditional_label), dim=2)
        tus_rank, tvs_rank, lus_rank, lvs_rank, les_rank =\
            next_tuple_dict['all_tuple_rank_dict'].values()
            
        sampled_ranks = torch.stack([tus_rank, tvs_rank, lus_rank, lvs_rank, les_rank],dim=1)
        tus, tvs, lus, lvs, les, = next_tuple_dict['all_tuple_value_dict'].values()
        
        return tus, tvs, lus, lvs, les, sampled_ranks

if __name__ == "__main__":
    print("cvae.py")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = argparse.ArgumentParser()
    parser = common_args(parser)
    args = parser.parse_args()
    params = Parameters(**utils.setup_params(vars(args), args.parameters))  # args，run_date，git_revisionなどを追加した辞書を取得
    
    model = CVAE(dfs_size=173, time_size=51, node_size=34, edge_size=2, condition_size=1, params=params, device="cuda")
    print(model)
    model.generate(data_num=int(300), conditional_label=torch.tensor([0.1]), max_size=int(357))
    print(model)
    