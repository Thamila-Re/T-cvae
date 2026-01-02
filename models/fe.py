"""
(C)VAEのDecoderから出力されたDFSコードの確率分布から、グラフ特徴量を算出するFeature estimatorモデルを定義するモジュール.
"""
import torch
from torch import nn


class FeatureEstimator(nn.Module):
    """Decoderの後ろに配置されるLSTM

    Decoderの後ろに配置されるLSTMであり、処理はEncoderと概ね同じである。
    DecodeされたDFSコードからグラフ特徴量を計算する。
    (モデル概要)
    線形層1(input_size, emb_sizae) => LSTM(emb_size, hidden_size) => (Optional) Dropout => 線形層2(hidden_size, rep_size)

    Functions
    ---------
    __init__() : LSTMとそのパラメータを定義する。
    forward()  : 順伝搬処理する。
    """
    def __init__(self, input_size:int, params:'config.Parameters', device:str) -> None:
        """コンストラクタ

        Args
        ----
        input_size                (int) : LSTMの前に配置される線形層へ入力するデータの次元数
        params      (config.Parameters) : configのglobal変数のset
        device           (torch.Device) : cpu or cuda device
        """
        super(FeatureEstimator, self).__init__()
        # 線形層1
        self.emb = nn.Linear(input_size, params.model_params["fe_emb_size"])
        # LSTM
        self.lstm = nn.LSTM(params.model_params["fe_emb_size"], params.model_params["fe_hidden_size"],
                            num_layers=params.model_params["fe_lstm_num_layer"], bidirectional=False, batch_first=True)
        # 線形層2
        self.calc_graph_property = nn.Linear(params.model_params["fe_hidden_size"], params.model_params["fe_rep_size"])
        # Dropout層
        self.dropout = nn.Dropout(params.model_params["fe_dropout"])
        # Loss function
        self.criterion = nn.MSELoss(reduction="sum")
        # Device info
        self.device = device
        # Acuuracy range
        self.acc_range = params.acc_range

    def forward(self, dfs_codes):
        """forwarding

        Args:
            dfs_codes (torch.tensor[batch_size, max_sequence_length, dfs_size-1]): onehotのDFSコードが格納されたミニバッチ

        Returns:
            (torch.tensor[rep_size]): ReEncoderのoutput
        """
        embedded_dfs_codes = self.emb(dfs_codes)
        output, (h,c) = self.lstm(embedded_dfs_codes)
        output = output[:, -1, :].unsqueeze(1)
        output = self.dropout(output)
        return self.calc_graph_property(output)

    def loss(self, results, targets):
        """modelの出力とCVAEのDecoderの出力との間でMSE lossを計算する関数.

        Args:
            results (torch.Tensor): modelの出力
            targets (torch.Tensor): CVAEのDecoderの出力分布から、Argmaxをとることで確定されたグラフ
                                    の特徴量

        Returns:
            (torch.Tensor): MSE loss
        """
        return self.criterion(results, targets)

    def loss_input_graph(self, results, targets):
        """modelの出力と入力グラフの統計量との間でMSE lossを計算する関数.

        Args:
            results (torch.Tensor): modelの出力
            targets (torch.Tensor): 入力グラフの統計量

        Returns:
            (torch.Tensor): MSE loss
        """
        results = results.transpose(1, 0)[0]
        targets = targets.transpose(1, 0)[0]
        return self.criterion(results, targets)

    def accuracy(self, results, targets, condition):
        """modelの出力精度を計算する関数

        Args:
            results (torch.Tensor): ReEncoderの出力値
            targets (torch.Tensor): グラフ特徴量のラベル
            condition (str): 精度を計算するグラフ特徴量の名前

        Returns:
            (float): 正解率(0 ~ 1.0)
        """
        results = results.transpose(1, 0)[0]
        targets = targets.transpose(1, 0)[0]
        correct_cnt = 0
        for pred, correct in zip(results, targets):
            # グラフ特徴量毎にラベルを基準にした正解と判定される範囲が事前定義されているので、その範囲内にあれば正確に予測されているとみなす
            if correct - self.acc_range[condition][0] <= pred and pred <= correct + self.acc_range[condition][1]:
                correct_cnt += 1
        return correct_cnt / len(results)


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from config import Parameters
    params = Parameters()
    model = FeatureEstimator(input_size=172, params=params, device="cpu")
    in_ = torch.randn([32,20,172])
    out_ = model(in_)
    # results = torch.Tensor([[1], [1.5], [2.0], [2.5], [3.0]])
    # targets = torch.Tensor([[2], [2], [2], [2], [2]])
    # model.accuracy(results, targets, params.condition_params[0])