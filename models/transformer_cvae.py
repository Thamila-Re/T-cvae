"""
Transformer-based Conditional Variational AutoEncoder (CVAE) implementation.
Fixed with proper sliding window attention and positional encoding for memory efficiency.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import math
import sys
import os
from collections import deque
from typing import Optional, Dict, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import try_gpu, sample_dist, convert2onehot, sample_topk_topp
from graph_process import graph_utils
from config import common_args, Parameters
import utils


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, position_offset=0):
        """Forward with optional position offset for sliding window"""
        if position_offset > 0:
            # Use offset for sliding window generation
            return x + self.pe[:, position_offset:position_offset + x.size(1), :]
        else:
            # Standard positional encoding
            return x + self.pe[:, :x.size(1), :]


class TransformerCVAE(nn.Module):
    """Transformer-based Conditional VAE class - matches LSTM CVAE interface"""
    
    def __init__(self, dfs_size, time_size, node_size, edge_size, condition_size, params, device):
        super(TransformerCVAE, self).__init__()
        
        # Model parameters - same structure as LSTM CVAE
        self.rep_size = params.model_params["rep_size"]
        self.alpha = params.model_params["alpha"]
        self.beta = params.model_params["beta"]
        self.word_drop = params.model_params["word_drop"]
        self.time_size = time_size
        self.node_size = node_size
        self.edge_size = edge_size
        self.device = device
        self.condition_size = condition_size
        
        # Transformer specific parameters
        d_model = params.model_params.get("transformer_d_model", 256)
        nhead = params.model_params.get("transformer_nhead", 8)
        num_encoder_layers = params.model_params.get("transformer_encoder_layers", 2)
        num_decoder_layers = params.model_params.get("transformer_decoder_layers", 2)
        dim_feedforward = params.model_params.get("transformer_dim_feedforward", 1024)
        dropout = params.model_params.get("transformer_dropout", 0.1)
        
        # Create encoder and decoder
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
                condition_size=self.condition_size, use_condition=params.model_params["encoder_condition"]
            )
        
        if params.model_params["decoder_h_c_condition"] is True and params.model_params["decoder_sequence_condition"] is False:
            self.decoder = TransformerDecoder(
                self.rep_size, dfs_size - condition_size, d_model, nhead, num_decoder_layers,
                dim_feedforward, time_size, node_size, edge_size, condition_size, params,
                dropout, self.device, h_c_condition=params.model_params["decoder_h_c_condition"], 
                seq_condition=params.model_params["decoder_sequence_condition"]
            )
        else:
            self.decoder = TransformerDecoder(
                self.rep_size, dfs_size, d_model, nhead, num_decoder_layers,
                dim_feedforward, time_size, node_size, edge_size, condition_size, params,
                dropout, self.device, h_c_condition=params.model_params["decoder_h_c_condition"], 
                seq_condition=params.model_params["decoder_sequence_condition"]
            )

    def forward(self, x, src_key_padding_mask=None):
        """Forward pass - NOW WITH PADDING MASK SUPPORT
        
        Args:
            x: Input tensor [batch, seq_len, features]
            src_key_padding_mask: Optional boolean mask [batch, seq_len] where True = padding
                                 If None, falls back to old behavior for backward compatibility
        """
        # Input validation
        assert x.dim() == 3, f"Expected x to be 3D, got {x.dim()}D"
        
        mu, sigma = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        z = self.transformation(mu, sigma, self.device)
        tu, tv, lu, lv, le = self.decoder(z, x, word_drop=self.word_drop)
        return mu, sigma, tu, tv, lu, lv, le

    @staticmethod
    def transformation(mu, sigma, device):
        """Reparametrization trick - same as LSTM CVAE"""
        return mu + torch.exp(0.5 * sigma) * torch.randn(sigma.shape).to(device)

    def loss(self, encoder_loss, decoder_loss):
        """Loss function - same as LSTM CVAE"""
        cvae_loss = self.beta * encoder_loss + self.alpha * decoder_loss
        return cvae_loss

    def generate(self, data_num, conditional_label, max_size, z=None, is_output_sampling=True):
        """Generate graph samples - same interface as LSTM CVAE"""
        if z is None:
            z = self.noise_generator(self.rep_size, data_num).unsqueeze(1)
            z = z.to(self.device)
        tu, tv, lu, lv, le = self.decoder.generate(z, conditional_label, max_size, is_output_sampling)
        return tu, tv, lu, lv, le

    def noise_generator(self, rep_size, batch_num):
        """Generate noise - same as LSTM CVAE"""
        return torch.randn(batch_num, rep_size)

    def generate_with_history(self, data_num: int, conditional_label: torch.Tensor, 
                              max_size: int, z: torch.Tensor=None):
        """Generate graph samples with history - same interface as LSTM CVAE"""
        if z is None:
            z = self.noise_generator(self.rep_size, data_num).unsqueeze(1)
            z = z.to(self.device)
        tu, tv, lu, lv, le, sampling_record = self.decoder.generate_with_history(z, conditional_label, max_size)
        return (tu, tv, lu, lv, le), sampling_record


class TransformerEncoder(nn.Module):
    """Transformer-based Encoder - FIXED to handle padding masks properly"""
    
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward, 
                 rep_size, dropout, device, condition_size, use_condition=True):
        super(TransformerEncoder, self).__init__()
        
        self.use_condition = use_condition
        self.condition_size = condition_size
        self.device = device
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projections for mu and sigma
        self.mu = nn.Linear(d_model, rep_size)
        self.sigma = nn.Linear(d_model, rep_size)

    def forward(self, x, src_key_padding_mask=None):
        """Forward pass - NOW WITH PADDING MASK SUPPORT
        
        Args:
            x: Input tensor [batch, seq_len, features]
            src_key_padding_mask: Optional boolean mask [batch, seq_len] where True = padding
                                 If None, falls back to old behavior (uses last token)
        """
        # Input validation
        assert x.dim() == 3, f"Expected x to be 3D, got {x.dim()}D"
        
        if self.use_condition is False:
            # Remove condition from input if not using it
            x = x[:, :, :-self.condition_size]
        
        # Project to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer WITH padding mask
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # FIXED: Use last non-padded token when mask is provided
        if src_key_padding_mask is not None:
            # Get sequence lengths (number of non-padded tokens)
            seq_lengths = (~src_key_padding_mask).sum(dim=1) - 1  
            
            # Clamp to valid range to avoid index errors
            seq_lengths = torch.clamp(seq_lengths, min=0, max=x.size(1) - 1)
            
            # Gather the last non-padded token for each sequence in batch
            batch_indices = torch.arange(x.size(0), device=x.device)
            x = x[batch_indices, seq_lengths].unsqueeze(1)
        else:
            
            x = x[:, -1, :].unsqueeze(1)
        
        return self.mu(x), self.sigma(x)

    def loss(self, mu, sigma):
        """KL divergence loss - same as LSTM Encoder"""
        delta = 1e-7
        return -0.5 * torch.sum(1 + sigma - mu ** 2 - torch.exp(sigma + delta))


class TransformerDecoder(nn.Module):
    """Transformer-based Decoder with sliding window for memory efficiency"""
    
    def __init__(self, rep_size, input_size, d_model, nhead, num_layers, dim_feedforward,
                 time_size, node_label_size, edge_label_size, condition_size, params, 
                 dropout, device, h_c_condition=True, seq_condition=True):
        super(TransformerDecoder, self).__init__()
        
        # Copy all parameters from LSTM decoder
        self.h_c_condition = h_c_condition
        self.seq_condition = seq_condition
        self.use_decoder_femb = params.use_decoder_femb
        self.h0c0_mode = params.h0c0_mode
        self.cat_rep_to_sos_and_input = params.cat_rep_to_sos_and_input
        self.sos_mode = params.sos_mode
        self.sampling_generation = params.sampling_generation
        self.condition_size = condition_size
        self.input_size = input_size
        self.d_model = d_model
        self.rep_size = rep_size
        self.device = device
        
        # Sliding window size for memory efficiency (corrected comment)
        self.window_size = 100  
        
        # Tuple names
        self.tuple_name = params.tuple_name
        
        # Calculate actual input size
        actual_input_size = input_size
        if self.use_decoder_femb:
            emb_input_size = actual_input_size
            transformer_input_size = d_model
        else:
            transformer_input_size = actual_input_size
        
        # Add rep size if concatenating
        if self.cat_rep_to_sos_and_input:
            if self.seq_condition:
                transformer_input_size += rep_size + condition_size
            else:
                transformer_input_size += rep_size
        
        # Input embedding
        if self.use_decoder_femb:
            self.emb = nn.Linear(emb_input_size, d_model)
        
        # SOS generation
        if self.sos_mode == 0:
            self.f_rep = nn.Linear(rep_size + condition_size, input_size)
        elif self.sos_mode == 2:
            self.f_rep = nn.Linear(condition_size, input_size)
        
        # Positional encoding with larger max_len for generation
        self.pos_encoder = PositionalEncoding(transformer_input_size, max_len=10000)
        
        # Memory projection
        if self.seq_condition:
            memory_input_size = rep_size + condition_size
        else:
            memory_input_size = rep_size
        # IMPORTANT: memory must be projected to the SAME size that decoder expects
        self.memory_projection = nn.Linear(memory_input_size, transformer_input_size)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=transformer_input_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projections
        self.f_tu = nn.Linear(transformer_input_size, time_size)
        self.f_tv = nn.Linear(transformer_input_size, time_size)
        self.f_lu = nn.Linear(transformer_input_size, node_label_size)
        self.f_lv = nn.Linear(transformer_input_size, node_label_size)
        self.f_le = nn.Linear(transformer_input_size, edge_label_size)
        
        self.f_tuple_dict = {
            self.tuple_name[0]: self.f_tu,
            self.tuple_name[1]: self.f_tv,
            self.tuple_name[2]: self.f_lu,
            self.tuple_name[3]: self.f_lv,
            self.tuple_name[4]: self.f_le
        }
        
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(0.5)
        
        # Additional parameters
        self.flag_softmax = params.args.get("softmax", True) if hasattr(params, 'args') else True
        self.time_size = time_size
        self.node_label_size = node_label_size
        self.edge_label_size = edge_label_size
        
        self.tuple_size_dict = {
            self.tuple_name[0]: self.time_size,
            self.tuple_name[1]: self.time_size,
            self.tuple_name[2]: self.node_label_size,
            self.tuple_name[3]: self.node_label_size,
            self.tuple_name[4]: self.edge_label_size
        }
        
        self.ignore_label = params.ignore_label
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_label, reduction="sum")
        
        # Sampling parameters
        self.softmax_temperature = params.softmax_temperature
        self.sampling_mode = getattr(params, 'sampling_mode', 'none')
        self.top_k = getattr(params, 'top_k', 10)
        self.top_p = getattr(params, 'top_p', 0.9)
        
        # Optional gradient checkpointing
        self.use_gradient_checkpointing = getattr(params, 'use_gradient_checkpointing', False)

    def _create_causal_mask(self, seq_len):
        """Create causal mask to prevent attention to future positions"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.to(self.device)

    def __calc_sos(self, rep, conditional, mode='train'):
        """Generate SOS token - same as LSTM decoder"""
        if self.sos_mode == 0:
            sos = self.f_rep(torch.cat((rep, conditional), dim=2))
        elif self.sos_mode == 1:
            sos = torch.cat([torch.zeros(conditional.shape[0], 1, self.input_size-1).to(self.device), conditional], dim=2)
        elif self.sos_mode == 2:
            sos = self.f_rep(conditional)
        else:
            raise ValueError("sos_mode is invalid.")
        return sos

    def __calc_next_tuple(self, x, all_tuple_dist_dict=None, all_tuple_value_dict=None,
                         all_tuple_rank_dict=None, mode=None):
        """Generate next tuple values - same logic as LSTM decoder"""
        if mode is None:
            raise ValueError("mode is None.")
        elif mode == "train":
            softmax_temperature = 1.0
        elif mode == "eval":
            softmax_temperature = self.softmax_temperature

        each_tuple_dist_dict = {}
        each_tuple_value_dict = {}
        each_tuple_rank_dict = {}

        # Initialize dictionaries if None - FIXED device initialization
        if all_tuple_dist_dict is None:
            all_tuple_dist_dict = {}
            for name in self.tuple_name:
                all_tuple_dist_dict[name] = torch.empty(0, device=self.device)
        
        if all_tuple_value_dict is None:
            all_tuple_value_dict = {}
            for name in self.tuple_name:
                all_tuple_value_dict[name] = torch.empty(0, dtype=torch.long, device=self.device)

        if all_tuple_rank_dict is None:
            all_tuple_rank_dict = {}
            for name in self.tuple_name:
                all_tuple_rank_dict[name] = torch.empty(0, dtype=torch.long, device=self.device)
        
        # Calculate distributions
        for name in self.tuple_name:
            each_tuple_dist_dict[name] = self.softmax(self.f_tuple_dict[name](x) / softmax_temperature)

        if mode == "train":
            next_tuple_dict = {
                'all_tuple_dist_dict': all_tuple_dist_dict,
                'all_tuple_value_dict': all_tuple_value_dict,
                'all_tuple_rank_dict': all_tuple_rank_dict,
                'each_tuple_dist_dict': each_tuple_dist_dict,
                'each_tuple_value_dict': each_tuple_value_dict,
                'each_tuple_rank_dict': each_tuple_rank_dict
            }
            return next_tuple_dict

        # For eval mode, sample values
        for name, dist in all_tuple_dist_dict.items():
            all_tuple_dist_dict[name] = torch.cat((dist, each_tuple_dist_dict[name]), dim=1)

        if self.sampling_mode != 'none':
            for name in self.tuple_name:
                each_tuple_value_dict[name], each_tuple_rank_dict[name] = sample_topk_topp(
                    each_tuple_dist_dict[name], k=self.top_k, p=self.top_p, mode=self.sampling_mode
                )
        elif self.sampling_generation:
            for name in self.tuple_name:
                each_tuple_value_dict[name], each_tuple_rank_dict[name] = sample_dist(each_tuple_dist_dict[name])
        else:
            for name in self.tuple_name:
                each_tuple_value_dict[name] = torch.argmax(each_tuple_dist_dict[name], dim=2)

        for name, tuple_value in all_tuple_value_dict.items():
            all_tuple_value_dict[name] = torch.cat((tuple_value, each_tuple_value_dict[name]), dim=1)

        for name, tuple_rank in all_tuple_rank_dict.items():
            all_tuple_rank_dict[name] = torch.cat((tuple_rank, each_tuple_rank_dict[name]), dim=1)

        for name, tuple_value in each_tuple_value_dict.items():
            # Use squeeze(1) to only remove the sequence dimension, preserving batch dimension
            one_hot = F.one_hot(tuple_value, self.tuple_size_dict[name])
            if one_hot.dim() == 3:  # [batch, seq=1, num_classes]
                each_tuple_value_dict[name] = one_hot.squeeze(1)  # Remove only seq dimension
            else:  # Already 2D
                each_tuple_value_dict[name] = one_hot

        next_tuple_dict = {
            'all_tuple_dist_dict': all_tuple_dist_dict,
            'all_tuple_value_dict': all_tuple_value_dict,
            'all_tuple_rank_dict': all_tuple_rank_dict,
            'each_tuple_dist_dict': each_tuple_dist_dict,
            'each_tuple_value_dict': each_tuple_value_dict,
            'each_tuple_rank_dict': each_tuple_rank_dict
        }
        
        return next_tuple_dict

    def forward(self, rep, x, word_drop=0):
        """Training forward pass with causal masking"""
        # Input validation
        assert rep.dim() == 3, f"Expected rep to be 3D, got {rep.dim()}D"
        assert x.dim() == 3, f"Expected x to be 3D, got {x.dim()}D"
        
        conditional_label = x[:, 0, -self.condition_size:].unsqueeze(1)

        if self.seq_condition:
            extended_rep = torch.cat([rep, conditional_label], dim=2)
        else:
            extended_rep = rep
        
        if self.seq_condition is False:
            x = x[:, :, :-self.condition_size]

        # Generate SOS
        sos = self.__calc_sos(rep, conditional_label, mode='train')
        x = torch.cat((sos, x), dim=1)[:, :-1, :]

        # Improved word dropout
        if word_drop > 0:
            # Create dropout mask efficiently
            dropout_mask = torch.bernoulli(
                torch.full((x.shape[0], x.shape[1], 1), 1 - word_drop, device=self.device)
            )
            # Apply mask
            if self.seq_condition:
                x[:, :, :-self.condition_size] *= dropout_mask
            else:
                x = x * dropout_mask

        # Project to model dimension if using embedding
        if self.use_decoder_femb:
            x = self.emb(x)
        
        # Concatenate rep to input if needed
        if self.cat_rep_to_sos_and_input:
            extended_rep_expanded = extended_rep.repeat(1, x.shape[1], 1)
            x = torch.cat((x, extended_rep_expanded), dim=2)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create memory
        memory = self.memory_projection(extended_rep)
        
        # Create causal mask
        tgt_mask = self._create_causal_mask(x.shape[1])
        
        # Transformer decoder with optional gradient checkpointing
        if self.training and self.use_gradient_checkpointing:
            from torch.utils.checkpoint import checkpoint
            # checkpoint requires a function; using lambda
            x = checkpoint(lambda a, b, c: self.transformer(a, b, tgt_mask=c), x, memory, tgt_mask)
        else:
            x = self.transformer(x, memory, tgt_mask=tgt_mask)
        
        x = self.dropout(x)

        # Calculate next tuples
        next_tuple_dict = self.__calc_next_tuple(x, mode='train')
        tu, tv, lu, lv, le = next_tuple_dict['each_tuple_dist_dict'].values()
        
        return tu, tv, lu, lv, le

    def generate(self, rep, conditional_label, max_size=100, is_output_sampling=True):
        """Generation with sliding window - memory efficient and preserves context"""
        conditional_label = conditional_label.repeat(rep.shape[0], 1, 1)

        if self.seq_condition:
            extended_rep = torch.cat([rep, conditional_label], dim=2)
        else:
            extended_rep = rep

        # Generate SOS
        sos = self.__calc_sos(rep, conditional_label, mode='eval')

        # Initialize dictionaries with proper device
        all_tuple_dist_dict = {name: torch.empty(0, device=self.device) for name in self.tuple_name}
        all_tuple_value_dict = {name: torch.empty(0, dtype=torch.long, device=self.device) for name in self.tuple_name}
        all_tuple_rank_dict = {name: torch.empty(0, dtype=torch.long, device=self.device) for name in self.tuple_name}

        # Create memory
        memory = self.memory_projection(extended_rep)
        
        # Use deque for efficient sliding window
        generated_tokens = deque(maxlen=self.window_size)
        current_token = sos
        
        # Track absolute position for proper positional encoding
        absolute_position = 0
        
        for i in range(max_size):
            # Embed current token
            if self.use_decoder_femb:
                token_emb = self.emb(current_token)
            else:
                token_emb = current_token
            
            # Add rep if concatenating
            if self.cat_rep_to_sos_and_input:
                token_emb = torch.cat((token_emb, extended_rep), dim=2)
            
            # Add to window
            generated_tokens.append(token_emb)
            
            # Stack window
            decoder_input = torch.cat(list(generated_tokens), dim=1)
            
            # Add positional encoding with proper offset
            position_offset = max(0, absolute_position - len(generated_tokens) + 1)
            decoder_input_pos = self.pos_encoder(decoder_input, position_offset=position_offset)
            
            # Create causal mask for window
            tgt_mask = self._create_causal_mask(decoder_input_pos.shape[1])
            
            # Process through transformer
            output = self.transformer(decoder_input_pos, memory, tgt_mask=tgt_mask)
            
            # Get last token
            x_last = output[:, -1:, :]

            # Calculate next tuple
            next_tuple_dict = self.__calc_next_tuple(
                x_last, 
                all_tuple_dist_dict=all_tuple_dist_dict,
                all_tuple_value_dict=all_tuple_value_dict,
                all_tuple_rank_dict=all_tuple_rank_dict,
                mode='eval'
            )

            # Get one-hot encoded values
            tu, tv, lu, lv, le = next_tuple_dict['each_tuple_value_dict'].values()
            # Ensure proper dimensions for concatenation
            if tu.dim() == 1:  # If batch dimension was squeezed out
                tu, tv, lu, lv, le = tu.unsqueeze(0), tv.unsqueeze(0), lu.unsqueeze(0), lv.unsqueeze(0), le.unsqueeze(0)
            current_token = torch.cat((tu, tv, lu, lv, le), dim=-1).unsqueeze(1).to(self.device)

            if self.seq_condition:
                current_token = torch.cat((current_token, conditional_label), dim=2)
            
            # Increment absolute position
            absolute_position += 1
            
            # Optional: Clear cache only when memory pressure is high
            if i % 200 == 0 and i > 0 and torch.cuda.is_available():
                if torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() > 0.9:
                    torch.cuda.empty_cache()

        if is_output_sampling:
            tus, tvs, lus, lvs, les = next_tuple_dict['all_tuple_value_dict'].values()
            return tus, tvs, lus, lvs, les
        else:
            tus_dist, tvs_dist, lus_dist, lvs_dist, les_dist = next_tuple_dict['all_tuple_dist_dict'].values()
            return tus_dist, tvs_dist, lus_dist, lvs_dist, les_dist

    def generate_with_history(self, rep, conditional_label, max_size=100):
        """Generation with history - using sliding window"""
        conditional_label = conditional_label.repeat(rep.shape[0], 1, 1)

        if self.seq_condition:
            extended_rep = torch.cat([rep, conditional_label], dim=2)
        else:
            extended_rep = rep

        # Generate SOS
        sos = self.__calc_sos(rep, conditional_label, mode='eval')

        # Initialize dictionaries with proper device
        all_tuple_dist_dict = {name: torch.empty(0, device=self.device) for name in self.tuple_name}
        all_tuple_value_dict = {name: torch.empty(0, dtype=torch.long, device=self.device) for name in self.tuple_name}
        all_tuple_rank_dict = {name: torch.empty(0, dtype=torch.long, device=self.device) for name in self.tuple_name}

        # Create memory
        memory = self.memory_projection(extended_rep)
        
        # Use deque for efficient sliding window
        generated_tokens = deque(maxlen=self.window_size)
        current_token = sos
        
        # Track absolute position
        absolute_position = 0
        
        for i in range(max_size):
            # Embed current token
            if self.use_decoder_femb:
                token_emb = self.emb(current_token)
            else:
                token_emb = current_token
            
            # Add rep if concatenating
            if self.cat_rep_to_sos_and_input:
                token_emb = torch.cat((token_emb, extended_rep), dim=2)
            
            # Add to window
            generated_tokens.append(token_emb)
            
            # Stack window
            decoder_input = torch.cat(list(generated_tokens), dim=1)
            
            
            position_offset = max(0, absolute_position - len(generated_tokens) + 1)
            decoder_input_pos = self.pos_encoder(decoder_input, position_offset=position_offset)
            
            # Create causal mask
            tgt_mask = self._create_causal_mask(decoder_input_pos.shape[1])
            
            # Process through transformer
            output = self.transformer(decoder_input_pos, memory, tgt_mask=tgt_mask)
            
            # Get last token
            x_last = output[:, -1:, :]

            # Calculate next tuple
            next_tuple_dict = self.__calc_next_tuple(
                x_last,
                all_tuple_dist_dict=all_tuple_dist_dict,
                all_tuple_value_dict=all_tuple_value_dict,
                all_tuple_rank_dict=all_tuple_rank_dict,
                mode='eval'
            )

            # Get one-hot encoded values
            tu, tv, lu, lv, le = next_tuple_dict['each_tuple_value_dict'].values()
            
            current_token = torch.cat((tu, tv, lu, lv, le), dim=-1).unsqueeze(1).to(self.device)

            if self.seq_condition:
                current_token = torch.cat((current_token, conditional_label), dim=2)
            
            # Increment absolute position
            absolute_position += 1
            
            # Conditional cache clearing
            if i % 200 == 0 and i > 0 and torch.cuda.is_available():
                if torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() > 0.9:
                    torch.cuda.empty_cache()

        tus_rank, tvs_rank, lus_rank, lvs_rank, les_rank = next_tuple_dict['all_tuple_rank_dict'].values()
        sampled_ranks = torch.stack([tus_rank, tvs_rank, lus_rank, lvs_rank, les_rank], dim=1)
        tus, tvs, lus, lvs, les = next_tuple_dict['all_tuple_value_dict'].values()
        
        return tus, tvs, lus, lvs, les, sampled_ranks

    def loss(self, results, targets):
        """Cross Entropy loss calculation - same as LSTM decoder"""
        total_loss = 0
        loss_dict = {}
        for i, (key, pred) in enumerate(results.items()):
            loss_dict[key] = self.criterion(pred.transpose(2, 1), targets[key])
            total_loss += loss_dict[key]
        return loss_dict.copy(), total_loss

    def accuracy(self, results, targets):
        """Classification accuracy calculation - same as LSTM decoder"""
        acc_dict = {}
        for i, (key, pred) in enumerate(results.items()):
            pred = torch.argmax(pred, dim=2)
            pred = pred.view(-1)
            targets[key] = targets[key].view(-1)
            score = torch.zeros(pred.shape[0])
            score[pred == targets[key]] = 1
            data_len = pred.shape[0]
            if self.ignore_label is not None:
                targets[key] = targets[key].cpu()
                ignore_args = np.where(targets[key] == self.ignore_label)[0]
                data_len -= len(ignore_args)
                score[ignore_args] = 0
            score = torch.sum(score) / data_len
            acc_dict[key] = score
        return acc_dict.copy()



def create_padding_mask_from_labels(label_tensor, ignore_label):
    """
    Create padding mask for transformer from label tensors.
    
    This function works with the labels created by your preprocessing (preprocess.py).
    Your preprocessing already marks padding positions with ignore_label, so we just
    need to convert that to a boolean mask.
    
    Args:
        label_tensor: [batch, seq_len] tensor with ignore_label for padding positions
                     (e.g., t_u_list, t_v_list, etc. from your preprocessing)
        ignore_label: The padding value used in preprocessing (e.g., -100)
        
    Returns:
        mask: [batch, seq_len] boolean tensor where True = padding position
    """
    return label_tensor == ignore_label


if __name__ == "__main__":
