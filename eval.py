"""
evaluation module that works with both LSTM and Transformer CVAE models..
"""

import os
from logging import getLogger
import argparse
import torch
import shutil
import joblib
import networkx as nx
import pickle
from pathlib import Path
from dataclasses import asdict
import json
import glob

from config import common_args, Parameters
from utils import setup_params, set_logging, get_condition_values, fix_seed, make_dir
from models import cvae, cvae_for_2_tuples
from models.transformer_cvae import TransformerCVAE  # Import the fixed Transformer CVAE
from models.hybrid_cvae import HybridCVAE  # Import the Hybrid CVAE if available
from graph_process import graph_utils
from graph_process.graph_statistic import GraphStatistic
from graph_process.graph_utils import random_graph_edit

logger = getLogger(__name__)

def eval(params: Parameters, eval_dir_name: str = "eval") -> None:
    """Evaluate trained CVAE model and generate graphs

    Args:
        params (config.Parameters): Global parameter set
        eval_dir_name (str): Evaluation result directory name (default: "eval")
    """

    # Set device
    device = torch.device(f"cuda:{params.gpu_id}" if torch.cuda.is_available() and params.gpu_id >= 0 else "cpu")
    logger.info(f"Using device: {device}")
    
    # Get result directory path
    if hasattr(params, 'result_dir') and params.result_dir:
        result_path = params.result_dir
        logger.info(f"Using specified result_dir: {result_path}")
    else:
        result_path = str(Path(params.args['eval_model']).parents[1])
        logger.info(f"Extracted result_path from model path: {result_path}")

    # Load preprocessed dataset parameters
    dataset_param_path = os.path.join(result_path, "dataset/param")
    logger.info(f"Looking for dataset parameters at: {dataset_param_path}")
    
    try:
        dataset_params = joblib.load(dataset_param_path)
        logger.info(f"Loaded {len(dataset_params)} parameters from dataset")
        logger.debug(f"Dataset params: {dataset_params}")
    
    # Flexible unpacking - handle different parameter counts
        if len(dataset_params) == 4:
            time_size, node_size, edge_size, conditional_size = dataset_params
            logger.info("Loaded 4 parameters (standard format)")
        elif len(dataset_params) == 5:
            time_size, node_size, edge_size, conditional_size, _ = dataset_params
            logger.info("Loaded 5 parameters, ignoring 5th parameter")
        elif len(dataset_params) == 6:
            time_size, node_size, edge_size, conditional_size, _, _ = dataset_params
            logger.info("Loaded 6 parameters, using first 4")
        else:
            logger.error(f"Unexpected parameter count: {len(dataset_params)}")
            logger.error(f"Parameters: {dataset_params}")
            raise ValueError(f"Cannot unpack {len(dataset_params)} parameters")
    
        logger.info("Successfully loaded dataset parameters")
    except FileNotFoundError:
        logger.error(f"Dataset parameters not found at: {dataset_param_path}")
        logger.error("Make sure you're using the correct --result_dir that matches your trained model")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset parameters: {str(e)}")
        raise

    logger.info("--------------")
    logger.info(f"time size: {time_size}")
    logger.info(f"node size: {node_size}")
    logger.info(f"edge size: {edge_size}")
    logger.info(f"conditional size: {conditional_size}")
    logger.info("--------------")

    # Function to check if generated graph is sufficient size
    is_sufficient_size = lambda graph: True if graph.number_of_nodes() > params.size_th else False

    # Model selection
    use_model = params.args['use_model']
    if use_model == 'cvae':
        dfs_size = 2 * time_size + 2 * node_size + edge_size + conditional_size
        model = cvae.CVAE(dfs_size, time_size, node_size, edge_size, conditional_size, params, device)
        logger.info("Using LSTM-based CVAE model")
    elif use_model == 'transformer_cvae':
        dfs_size = 2 * time_size + 2 * node_size + edge_size + conditional_size
        model = TransformerCVAE(dfs_size, time_size, node_size, edge_size, conditional_size, params, device)
        logger.info("Using Transformer-based CVAE model")
    elif use_model == 'hybrid_cvae':
        dfs_size = 2 * time_size + 2 * node_size + edge_size + conditional_size
        model = HybridCVAE(dfs_size, time_size, node_size, edge_size, conditional_size, params, device)
        logger.info("Using Hybrid CVAE model")
    elif use_model == 'cvae_for_2_tuples':
        dfs_size = 2 * time_size + conditional_size
        model = cvae_for_2_tuples.CVAE(dfs_size, time_size, conditional_size, params, device)
        logger.info("Using LSTM-based CVAE for 2-tuples model")
    else:
        logger.error("Model not selected!")
        logger.error(f"Supported models: cvae, transformer_cvae, hybrid_cvae, cvae_for_2_tuples")
        logger.error(f"Provided model: {use_model}")
        exit()
    logger.info(f"Use model: {use_model}")

    # Load trained model
    if params.args['eval_model']:
        logger.info("Model Path : {}".format(params.args['eval_model']))
        checkpoint = torch.load(params.args['eval_model'], map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        model = model.to(device)
    eval_dir = os.path.join(result_path, eval_dir_name)
    os.makedirs(eval_dir, exist_ok=True)
    logger.info("Model loaded")

    # Create conditions for conditional generation
    conditional_labels = get_condition_values(params.condition_params, params.condition_values)

    # Label normalization (normalize by train data scale)
    if params.normalize:
        logger.info("Normalize conditional labels scaling by train data")
        datasets = joblib.load(params.twitter_train_path)
        train_label = datasets[1]
        # Search max, min
        max_val, min_val = -1, 10000
        for i, label in enumerate(train_label):
            if max_val < label.item():
                max_val = label.item()
            if min_val > label.item():
                min_val = label.item()
        # Normalization
        for i, data in enumerate(conditional_labels):
            conditional_labels[i] = (data - min_val) / (max_val - min_val)

    # Handle DFS history recording if specified
    record_size = params.args['dfs_history']
    if record_size is not None:
        logger.info("Get the history of generated graph")
        
        # For Transformer CVAE, generate history one at a time to avoid OOM
        if use_model == 'transformer_cvae':
            logger.info("Generating DFS history one sample at a time for Transformer CVAE")
            recorded_result_low, record_low = model.generate_with_history(
                1,
                torch.tensor([conditional_labels[0]]).float().to(device),
                max_size=params.generate_edge_num
            )
            torch.cuda.empty_cache()
            
            recorded_result_middle, record_middle = model.generate_with_history(
                1,
                torch.tensor([conditional_labels[1]]).float().to(device),
                max_size=params.generate_edge_num
            )
            torch.cuda.empty_cache()
            
            recorded_result_high, record_high = model.generate_with_history(
                1,
                torch.tensor([conditional_labels[2]]).float().to(device),
                max_size=params.generate_edge_num
            )
            torch.cuda.empty_cache()
        else:
            # Generate with history for each condition (original code for other models)
            recorded_result_low, record_low = model.generate_with_history(
                record_size,
                torch.tensor([conditional_labels[0]]).float().to(device),
                max_size=params.generate_edge_num
            )
            recorded_result_middle, record_middle = model.generate_with_history(
                record_size,
                torch.tensor([conditional_labels[1]]).float().to(device),
                max_size=params.generate_edge_num
            )
            recorded_result_high, record_high = model.generate_with_history(
                record_size,
                torch.tensor([conditional_labels[2]]).float().to(device),
                max_size=params.generate_edge_num
            )
        
        recorded_dfs_all = [recorded_result_low, recorded_result_middle, recorded_result_high]
        record_all = [record_low, record_middle, record_high]
        
        # Save generated DFS codes and sampling records
        records_dir = os.path.join(eval_dir, "dfs_history")
        os.makedirs(records_dir, exist_ok=True)
        torch.save(recorded_dfs_all, os.path.join(records_dir, "dfs_code.pt"))
        torch.save(record_all, os.path.join(records_dir, "sampled_rank.pt"))
        c_label = "\n".join([str(_) for _ in conditional_labels])
        with open(os.path.join(records_dir, "condition.txt"), 'w') as f:
            f.write(c_label)

    logger.info('Start generating graphs')
    
    # Clear GPU cache before generation
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        logger.info(f"GPU memory before generation: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Generate graphs for each condition
    # For Transformer CVAE, use sequential one-at-a-time generation to avoid OOM
    if use_model in ['transformer_cvae', 'hybrid_cvae']:
        logger.info("Using sequential one-at-a-time generation for Transformer CVAE to avoid OOM")
        result_all = []
        
        for cond_idx, cond_label in enumerate(conditional_labels):
            logger.info(f"Generating for condition {cond_idx+1}/3: {cond_label}")
            all_tu, all_tv, all_lu, all_lv, all_le = [], [], [], [], []
            
            for i in range(params.number_of_generated_samples):
                if i % 50 == 0:
                    logger.info(f"  Progress: {i}/{params.number_of_generated_samples} graphs generated")
                
                try:
                    tu, tv, lu, lv, le = model.generate(
                        1,  # Generate ONE graph at a time
                        torch.tensor([cond_label]).float().to(device),
                        max_size=params.generate_edge_num,
                    )
                    
                    all_tu.append(tu)
                    all_tv.append(tv)
                    all_lu.append(lu)
                    all_lv.append(lv)
                    all_le.append(le)
                    
                    # Clear GPU cache every 10 graphs
                    if i % 10 == 0 and device.type == 'cuda':
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logger.error(f"Error generating graph {i} for condition {cond_idx}: {str(e)}")
                    raise
            
            # Concatenate all generated graphs for this condition
            result = (
                torch.cat(all_tu, dim=0),
                torch.cat(all_tv, dim=0),
                torch.cat(all_lu, dim=0),
                torch.cat(all_lv, dim=0),
                torch.cat(all_le, dim=0)
            )
            result_all.append(result)
            logger.info(f"Completed generation for condition {cond_idx+1}/3")
    else:
        # Original code for LSTM CVAE and other models
        try:
            result_low = model.generate(
                params.number_of_generated_samples,
                torch.tensor([conditional_labels[0]]).float().to(device),
                max_size=params.generate_edge_num,
            )
            result_middle = model.generate(
                params.number_of_generated_samples,
                torch.tensor([conditional_labels[1]]).float().to(device),
                max_size=params.generate_edge_num,
            )
            result_high = model.generate(
                params.number_of_generated_samples,
                torch.tensor([conditional_labels[2]]).float().to(device),
                max_size=params.generate_edge_num,
            )
            result_all = [result_low, result_middle, result_high]
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            raise

    # Process generated results
    for index, (result, cond_label) in enumerate(zip(result_all, conditional_labels)):
        # Convert generated DFS codes to graphs
        result = [code.unsqueeze(2) for code in result]
        dfs_code = torch.cat(result, dim=2)
        generated_graph = []
        
        for code in dfs_code:
            try:
                graph = graph_utils.dfs_code_to_graph_obj(
                    code.cpu().detach().numpy(),
                    [time_size, time_size, node_size, node_size, edge_size],
                    edge_num=params.generate_edge_num
                )
                if is_sufficient_size(graph):
                    generated_graph.append(graph)
            except Exception as e:
                logger.debug(f"Failed to convert DFS code to graph: {e}")

        # Save generated graphs
        output_file = os.path.join(eval_dir, f"{'_'.join(params.condition_params)}_{cond_label}.pkl")
        joblib.dump(generated_graph, output_file)
        logger.info(f"Saved {len(generated_graph)} graphs to {output_file}")
    
    logger.info('Generated graphs are saved')
    
    # Save randomly edited train graphs if specified
    if params.args['edit_train_graph']:
        logger.info('Start editing train graphs')
        # Create GraphStatistic object
        graph_statistic = GraphStatistic(params)
        train_graph_files = graph_statistic.get_graph()[params.args['dataset']+'_train']
        logger.info(f'Number of graph files: {len(train_graph_files)}')
        logger.debug(f'Check type: {type(train_graph_files[0])}')
        
        # Edit graphs randomly
        train_graphs = []
        for i, graph in enumerate(train_graph_files):
            train_graph = random_graph_edit(graph, params.del_ratio)
            train_graphs.append(train_graph)
        logger.debug(f'Number of train graphs: {len(train_graphs)}')
        assert len(train_graphs) == len(train_graph_files)
        
        # Save graphs
        assert Path(eval_dir).exists(), f'{eval_dir} does not exist'
        with open(os.path.join(eval_dir, 'train_graph_random_edit.pkl'), 'wb') as f:
            pickle.dump(train_graphs, f)
        logger.info('Edited train graphs are saved')
    
    logger.info("Evaluation is completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = common_args(parser)
    args = parser.parse_args()
    params = Parameters(**setup_params(vars(args), args.parameters))

    if params.args['eval_model'] is None:
        logger.info("Please specify the model to evaluate.")
        exit()

    # Setup logging
    result_dir = params.result_dir
    logger = getLogger(__name__)

    # Run evaluation
    set_logging(result_dir, file_name="eval")
    # Fix seed
    seed = params.seed
    logger.info(f"Fixed seed to {seed}")
    fix_seed(seed)
    eval(params)