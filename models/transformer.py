import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer():
    pass

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, head_size, droprate):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size, 1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, head_size)
        self.attn_dropout = nn.Dropout(droprate)

        self.ffn_norm = nn.LayerNorm(hidden_size, 1e-6)
        self.ffn = PositionalwiseFeedForward(hidden_size, filter_size)
        self.ffn_dropout = nn.Dropout(droprate)

    def forward(self, x, mask):
        y = self.attn_norm(x)
        y = self.self_attention(y, y, y, mask)
        y = self.attn_dropout(y)
        x = x + y # Residual connection

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y # Residual connection

        return x

class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, head_size, droprate):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size, 1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, head_size)
        self.attn_dropout = nn.Dropout(droprate)

        self.enc_dec_attn_norm = nn.LayerNorm(hidden_size, 1e-6)
        self.enc_dec_attention = MultiHeadAttention(hidden_size, head_size)
        self.enc_dec_dropoout = nn.Dropout(droprate)

        self.ffn_norm = nn.LayerNorm(hidden_size, 1e-6)
        self.ffn = PositionalwiseFeedForward(hidden_size, filter_size)
        self.ffn_dropout = nn.Dropout(droprate)

    def forward(self, x, enc_output, self_mask, i_mask):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_
        enc_output : _type_
            _description_
        self_mask : _type_
            おそらくpaddingのためのmask
        i_mask : _type_
            おそらくdecoderで情報がleakしないためのmask

        Returns
        -------
        _type_
            _description_
        """
        y = self.attn_norm(x)
        y = self.self_attention(y, y, y, self_mask)
        y = self.attn_dropout(y)
        x = x + y # Residual connection

        y = self.enc_dec_attn_norm(x)
        y = self.enc_dec_attn_norm(y, enc_output, enc_output, i_mask)
        y = self.enc_dec_dropoout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y # Residual connection

        return x

def initialize_weight(x):
    """線形層の重みをxavierで初期化する関数"""
    nn.init.xavier_normal_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


class PositionalwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(hidden_size, filter_size) # position-wise
        self.w_2 = nn.Linear(filter_size, hidden_size) # position-wise
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.w_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, head_size, droprate=0.4, d_k=None, d_v=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_size = head_size

        self.attr_size = hidden_size // head_size

        if d_k is not None:
            self.d_k = d_k
        else:
            self.d_k = self.attr_size
        
        if d_v is not None:
            self.d_v = d_v
        else:
            self.d_v = self.attr_size

        self.w_q = nn.Linear(hidden_size, self.attr_size * head_size, bias=False)
        self.w_k = nn.Linear(hidden_size, self.attr_size * head_size, bias=False)
        self.w_v = nn.Linear(hidden_size, self.attr_size * head_size, bias=False)
        initialize_weight(self.w_q)
        initialize_weight(self.w_k)
        initialize_weight(self.w_v)

        self.attention = ScaledDotProductAttention(self.d_k, droprate)

        self.output_layer = nn.Linear(head_size * self.att_size, hidden_size,bias=False)
        initialize_weight(self.output_layer)

    def forward(self, q, k, v, mask):

        # transformerは複数レイヤーを繰り返すため、入力と出力のshapeは同じである必要がある
        origin_shape = q.size()

        batch_size = q.size(0)

        # 入力値をheadの数に対応するように分割する
        # shapeは[batch_size, seq_len, head_size, d_(k|v)]
        q = self.w_q(q).view(batch_size, -1, self.head_size, self.d_k)
        k = self.w_k(k).view(batch_size, -1, self.head_size, self.d_k)
        v = self.w_v(v).view(batch_size, -1, self.head_size, self.d_v)

        # 計算の整合性を保つためにtransposeする
        # headとseq_lenを入れ替える
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        # この時点でのshapeは[batch, head, seq_len, d_v]
        x = self.attention(q,k,v, mask)

        # 一度shapeを[batch_size, seq_len, head_size, d_v]に戻す
        x = x.transpose(1,2).contiguous()

        # headをconcatする
        x = x.view(batch_size, -1, self.head_size * self.d_v)

        x = self.output_layer(x)

        # この層を何個も繰り返すため、入力と出力のshapeは同じである必要がある
        assert x.size() == origin_shape
        return x
        
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, attn_dropout=0.4):
        super().__init__()

        self.d_k = d_k
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.d_k, k.transpose(2,3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

if __name__=='__main__':
    test_q = torch.randn(2,4,6)
    test_k = torch.randn(2,4,6)
    test_v = torch.randn(2,4,6)

    mha = MultiHeadAttention(6, 3)
    mha(test_q)

    attention = ScaledDotProductAttention(5**0.5)

    output, attn = attention(test_q, test_k, test_v)