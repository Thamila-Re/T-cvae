"""
Hybrid Conditional Variational AutoEncoder (CVAE) implementation.
Supports two configurations:
1. sequence_model='transformer': Transformer Encoder + LSTM Decoder
2. sequence_model='lstm': LSTM Encoder + Transformer Decoder
"""

import torch
from torch import nn
import sys
import os
from logging import getLogger

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import Parameters

# Import LSTM components
from models.cvae import Encoder as LSTMEncoder, Decoder as LSTMDecoder

# Import Transformer components
from models.transformer_cvae import TransformerEncoder, TransformerDecoder

logger = getLogger(__name__)


class HybridCVAE(nn.Module):
    """Hybrid Conditional VAE class - NOW WITH PADDING MASK SUPPORT
    
    Supports two configurations:
    - sequence_model='transformer': Transformer Encoder + LSTM Decoder
    - sequence_model='lstm': LSTM Encoder + Transformer Decoder
    """
    
    def __init__(self, dfs_size, time_size, node_size, edge_size, condition_size, params, device):
        super(HybridCVAE, self).__init__()
        
        self.rep_size = params.model_params["rep_size"]
        self.alpha = params.model_params["alpha"]
        self.beta = params.model_params["beta"]
        self.word_drop = params.model_params["word_drop"]
        self.time_size = time_size
        self.node_size = node_size
        self.edge_size = edge_size
        self.device = device
        self.condition_size = condition_size
        
        # Get sequence model configuration
        self.sequence_model = params.model_params.get("sequence_model", "transformer")
        
        if self.sequence_model not in ["transformer", "lstm"]:
            raise ValueError(f"sequence_model must be 'transformer' or 'lstm', got {self.sequence_model}")
        
        # Create encoder and decoder based on sequence_model
        if self.sequence_model == "transformer":
            # Configuration 1: Transformer Encoder + LSTM Decoder
            print(f"\n{'='*60}")
            print(f"Initializing Hybrid CVAE: Transformer Encoder + LSTM Decoder")
            print(f"{'='*60}\n")
            
            # Transformer Encoder parameters
            d_model = params.model_params.get("transformer_d_model", 256)
            nhead = params.model_params.get("transformer_nhead", 8)
            num_encoder_layers = params.model_params.get("transformer_encoder_layers", 2)
            dim_feedforward = params.model_params.get("transformer_dim_feedforward", 1024)
            dropout = params.model_params.get("transformer_dropout", 0.1)
            
            print(f"Transformer Encoder Config:")
            print(f"  - d_model: {d_model}")
            print(f"  - nhead: {nhead}")
            print(f"  - num_layers: {num_encoder_layers}")
            print(f"  - dim_feedforward: {dim_feedforward}")
            print(f"  - dropout: {dropout}\n")
            
            # Create Transformer Encoder
            if params.model_params["encoder_condition"]:
                self.encoder = TransformerEncoder(
                    dfs_size, d_model, nhead, num_encoder_layers, dim_feedforward,
                    self.rep_size, dropout, self.device, condition_size=self.condition_size,
                    use_condition=params.model_params["encoder_condition"]
                )
            else:
                self.encoder = TransformerEncoder(
                    dfs_size - self.condition_size, d_model, nhead, num_encoder_layers,
                    dim_feedforward, self.rep_size, dropout, self.device,
                    condition_size=self.condition_size,
                    use_condition=params.model_params["encoder_condition"]
                )
            
            # LSTM Decoder parameters
            emb_size = params.model_params["emb_size"]
            de_hidden_size = params.model_params["de_hidden_size"]
            
            print(f"LSTM Decoder Config:")
            print(f"  - emb_size: {emb_size}")
            print(f"  - hidden_size: {de_hidden_size}\n")
            
            # Create LSTM Decoder
            if params.model_params["decoder_h_c_condition"] is True and params.model_params["decoder_sequence_condition"] is False:
                self.decoder = LSTMDecoder(
                    self.rep_size, dfs_size - condition_size, emb_size, de_hidden_size,
                    time_size, node_size, edge_size, condition_size, params, self.device,
                    h_c_condition=params.model_params["decoder_h_c_condition"],
                    seq_condition=params.model_params["decoder_sequence_condition"]
                )
            else:
                self.decoder = LSTMDecoder(
                    self.rep_size, dfs_size, emb_size, de_hidden_size,
                    time_size, node_size, edge_size, condition_size, params, self.device,
                    h_c_condition=params.model_params["decoder_h_c_condition"],
                    seq_condition=params.model_params["decoder_sequence_condition"]
                )
                
        else:  # sequence_model == "lstm"
            # Configuration 2: LSTM Encoder + Transformer Decoder
            print(f"\n{'='*60}")
            print(f"Initializing Hybrid CVAE: LSTM Encoder + Transformer Decoder")
            print(f"{'='*60}\n")
            
            # LSTM Encoder parameters
            emb_size = params.model_params["emb_size"]
            en_hidden_size = params.model_params["en_hidden_size"]
            
            print(f"LSTM Encoder Config:")
            print(f"  - emb_size: {emb_size}")
            print(f"  - hidden_size: {en_hidden_size}\n")
            
            # Create LSTM Encoder
            if params.model_params["encoder_condition"]:
                self.encoder = LSTMEncoder(
                    dfs_size, emb_size, en_hidden_size, self.rep_size, self.device,
                    condition_size=self.condition_size,
                    use_condition=params.model_params["encoder_condition"]
                )
            else:
                self.encoder = LSTMEncoder(
                    dfs_size - self.condition_size, emb_size, en_hidden_size,
                    self.rep_size, self.device, condition_size=self.condition_size,
                    use_condition=params.model_params["encoder_condition"]
                )
            
            # Transformer Decoder parameters
            d_model = params.model_params.get("transformer_d_model", 256)
            nhead = params.model_params.get("transformer_nhead", 8)
            num_decoder_layers = params.model_params.get("transformer_decoder_layers", 2)
            dim_feedforward = params.model_params.get("transformer_dim_feedforward", 1024)
            dropout = params.model_params.get("transformer_dropout", 0.1)
            
            print(f"Transformer Decoder Config:")
            print(f"  - d_model: {d_model}")
            print(f"  - nhead: {nhead}")
            print(f"  - num_layers: {num_decoder_layers}")
            print(f"  - dim_feedforward: {dim_feedforward}")
            print(f"  - dropout: {dropout}\n")
            
            # Create Transformer Decoder
            if params.model_params["decoder_h_c_condition"] is True and params.model_params["decoder_sequence_condition"] is False:
                self.decoder = TransformerDecoder(
                    self.rep_size, dfs_size - condition_size, d_model, nhead,
                    num_decoder_layers, dim_feedforward, time_size, node_size,
                    edge_size, condition_size, params, dropout, self.device,
                    h_c_condition=params.model_params["decoder_h_c_condition"],
                    seq_condition=params.model_params["decoder_sequence_condition"]
                )
            else:
                self.decoder = TransformerDecoder(
                    self.rep_size, dfs_size, d_model, nhead, num_decoder_layers,
                    dim_feedforward, time_size, node_size, edge_size,
                    condition_size, params, dropout, self.device,
                    h_c_condition=params.model_params["decoder_h_c_condition"],
                    seq_condition=params.model_params["decoder_sequence_condition"]
                )
        
        print(f"{'='*60}")
        print(f"Hybrid CVAE initialization complete!")
        print(f"Configuration: sequence_model='{self.sequence_model}'")
        print(f"Padding mask support: {'Enabled (Transformer Encoder)' if self.sequence_model == 'transformer' else 'N/A (LSTM Encoder)'}")
        print(f"{'='*60}\n")

    def forward(self, x, src_key_padding_mask=None):
        """Forward pass - NOW WITH PADDING MASK SUPPORT
        
        Args:
            x: Input tensor [batch, seq_len, features]
            src_key_padding_mask: Optional boolean mask [batch, seq_len] where True = padding
                                 Only used when sequence_model='transformer'
                                 Ignored for LSTM encoder (backward compatible)
        """
        assert x.dim() == 3, f"Expected x to be 3D, got {x.dim()}D"
        
        # Pass mask to encoder only if using Transformer Encoder
        if self.sequence_model == "transformer":
            # Transformer Encoder - pass padding mask
            mu, sigma = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        else:
            # LSTM Encoder - doesn't use padding mask, backward compatible
            mu, sigma = self.encoder(x)
        
        z = self.transformation(mu, sigma, self.device)
        tu, tv, lu, lv, le = self.decoder(z, x, word_drop=self.word_drop)
        return mu, sigma, tu, tv, lu, lv, le

    @staticmethod
    def transformation(mu, sigma, device):
        """Reparametrization trick - same as other CVAE models"""
        return mu + torch.exp(0.5 * sigma) * torch.randn(sigma.shape).to(device)

    def loss(self, encoder_loss, decoder_loss):
        """Loss function - same as other CVAE models"""
        cvae_loss = self.beta * encoder_loss + self.alpha * decoder_loss
        return cvae_loss

    def generate(self, data_num, conditional_label, max_size, z=None, is_output_sampling=True):
        """Generate graph samples - same interface as other CVAE models"""
        
        original_data_num = data_num
        if data_num == 1 and self.sequence_model == "transformer":
            data_num = 2
            logger.debug(f"Increasing batch_size from 1 to 2 to avoid LSTM decoder shape issues")
        
        if z is None:
            z = self.noise_generator(self.rep_size, data_num)
            z = z.to(self.device)
        
        # BOTH decoders expect 3D z: [batch, 1, rep_size]
        if z.dim() == 2:
            z = z.unsqueeze(1)
        
        if self.sequence_model == "transformer":
            # LSTM Decoder expects [1, 1, condition_size]
            if conditional_label.dim() == 0:
                conditional_label = conditional_label.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            elif conditional_label.dim() == 1:
                conditional_label = conditional_label.unsqueeze(0).unsqueeze(0)
            elif conditional_label.dim() == 2:
                conditional_label = conditional_label[0:1, :].unsqueeze(1)
            elif conditional_label.dim() == 3:
                conditional_label = conditional_label[0:1, 0:1, :]
            
            if conditional_label.size(0) != 1 or conditional_label.size(1) != 1:
                conditional_label = conditional_label[0:1, 0:1, :]
        else:
            # Transformer Decoder expects [batch, 1, condition_size]
            if conditional_label.dim() == 0:
                conditional_label = conditional_label.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            elif conditional_label.dim() == 1:
                conditional_label = conditional_label.unsqueeze(0).unsqueeze(0)
            elif conditional_label.dim() == 2:
                conditional_label = conditional_label.unsqueeze(1)
            
            if conditional_label.size(0) == 1 and data_num > 1:
                conditional_label = conditional_label.repeat(data_num, 1, 1)
        
        try:
            tu, tv, lu, lv, le = self.decoder.generate(z, conditional_label, max_size, is_output_sampling)
        except Exception as e:
            logger.error(f"Error in decoder.generate:")
            logger.error(f"  z shape: {z.shape}")
            logger.error(f"  conditional_label shape: {conditional_label.shape}")
            logger.error(f"  data_num: {data_num}")
            logger.error(f"  max_size: {max_size}")
            raise e
        
        # Restore original batch size if we increased it
        if original_data_num != data_num:
            tu = tu[:original_data_num]
            tv = tv[:original_data_num]
            lu = lu[:original_data_num]
            lv = lv[:original_data_num]
            le = le[:original_data_num]
        
        # Ensure outputs are 2D
        def ensure_2d(tensor):
            if tensor.dim() == 1:
                return tensor.unsqueeze(0)  
            elif tensor.dim() == 0:
                return tensor.unsqueeze(0).unsqueeze(0)  
            return tensor
        
        tu = ensure_2d(tu)
        tv = ensure_2d(tv)
        lu = ensure_2d(lu)
        lv = ensure_2d(lv)
        le = ensure_2d(le)
        
        return tu, tv, lu, lv, le

    def noise_generator(self, rep_size, batch_num):
        """Generate noise - same as other CVAE models"""
        return torch.randn(batch_num, rep_size)

    def generate_with_history(self, data_num: int, conditional_label: torch.Tensor,
                              max_size: int, z: torch.Tensor=None):
        """Generate graph samples with history - wrapper that handles shape consistency"""
        if z is None:
            z = self.noise_generator(self.rep_size, data_num)
            z = z.to(self.device)
        
        # Handle z shape based on decoder type
        if self.sequence_model == "transformer":
            if z.dim() == 2:
                z = z.unsqueeze(1)
        else:
            if z.dim() == 3 and z.size(1) == 1:
                z = z.squeeze(1)
        
        # Handle conditional_label shape
        if conditional_label.dim() == 1:
            conditional_label = conditional_label.unsqueeze(-1)  # [batch] -> [batch, 1]
        elif conditional_label.dim() == 0:
            conditional_label = conditional_label.unsqueeze(0).unsqueeze(0)  # [] -> [1, 1]
        elif conditional_label.dim() == 3:
            conditional_label = conditional_label.squeeze(1)  # [batch, 1, cond] -> [batch, cond]
        
        # Replicate if needed
        if conditional_label.size(0) == 1 and data_num > 1:
            conditional_label = conditional_label.repeat(data_num, 1)
        
        # Validate batch size
        if conditional_label.size(0) != data_num:
            if conditional_label.size(0) == 1:
                conditional_label = conditional_label.repeat(data_num, 1)
            else:
                raise ValueError(f"conditional_label batch size {conditional_label.size(0)} doesn't match data_num {data_num}")
        
        # Call decoder's generate_with_history
        try:
            tu, tv, lu, lv, le, sampling_record = self.decoder.generate_with_history(z, conditional_label, max_size)
        except Exception as e:
            logger.error(f"Error in decoder.generate_with_history:")
            logger.error(f"  z shape: {z.shape}")
            logger.error(f"  conditional_label shape: {conditional_label.shape}")
            logger.error(f"  data_num: {data_num}")
            logger.error(f"  max_size: {max_size}")
            raise e
        
        # Ensure outputs are 2D
        def ensure_2d(tensor):
            if tensor.dim() == 1:
                return tensor.unsqueeze(0)
            elif tensor.dim() == 0:
                return tensor.unsqueeze(0).unsqueeze(0)
            return tensor
        
        tu = ensure_2d(tu)
        tv = ensure_2d(tv)
        lu = ensure_2d(lu)
        lv = ensure_2d(lv)
        le = ensure_2d(le)
        
        return (tu, tv, lu, lv, le), sampling_record


if __name__ == "__main__":
    
