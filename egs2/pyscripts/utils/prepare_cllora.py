import argparse
import logging
from pathlib import Path
import torch
import yaml

import assign_lora_r as alr

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Collect arguments for the model configuration and loading.")
    
    allowed_blocks = ["linear_q", "linear_k", "linear_v", "linear_out", # attention-related
                      "channel_proj1", "channel_proj2", "feed_forward", "feed_forward_macaron", "merge_proj"] # decoder token related

    strategy = [
            "attention", # "linear_q", "linear_k", "linear_v"
            "attention+", # "linear_q", "linear_k", "linear_v", "linear_out"
            "attention + feedforward", # "linear_q", "linear_k", "linear_v", "linear_out", "feed_forward", "feed_forward_macaron"
            "attention + feedforward + output", # "linear_q", "linear_k", "linear_v", "linear_out", "feed_forward", "feed_forward_macaron", "output_layer", "embed"
            "all", # entire list above
    ]

    parser.add_argument("--model_dir", type=str, default="exp/ml/s2t_train_asr_owsmv31_base_raw_pretrained_bpe50000", help="Path to the directory of the model.")
    parser.add_argument("--model_name", type=str, default="valid.acc.ave.pth", help="Name of the model file (stored with PyTorch).")
    parser.add_argument("--prev_yaml_file", type=str, default="conf/models/whisper/cl/lora/owsm_to_cgnnl/train_asr_owsmv31_base_cllora_r8_default.yaml", 
                                         help="Path and file name of the previous yaml file configuration.")
    parser.add_argument("--blocks", type=str, choices=strategy, default="all",
                        help="Layer types to be used. Allowed values are: " + ", ".join(strategy))
    parser.add_argument('--r', type=int, default=8, help="Size of the lora adapters")
    parser.add_argument('--lang_sym', type=int, default=93, help="ID of the Language")
    parser.add_argument('--task_label', type=int, default=0, help="ID of the task")
    parser.add_argument("yaml_name", type=str, help="Path and name of the to-be-generated yaml file.")
    
    return parser.parse_args()

def prepare_paths(args):
    """Prepare paths for model configuration and model file."""
    model_dir = Path(args.model_dir)
    model_config_path = model_dir / "config.yaml"
    model_file_path = model_dir / args.model_name
    
    return model_config_path, model_file_path

def load_yaml(file_path):
    """Load YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def save_yaml(data, file_path):
    """Save YAML file."""
    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file)

def update_yaml_config(yaml_data, lora_layers, lang_sym, task_label):
    """Update YAML configuration with LoRA layers."""
    # Ensure use_adapter is set to True
    if 'use_adapter' not in yaml_data:
        yaml_data['use_adapter'] = True

    # Ensure adapter is set to cllora
    if 'adapter' not in yaml_data:
        yaml_data['adapter'] = 'cllora'

    # Ensure adapter_conf and target_blocks are set
    if 'adapter_conf' not in yaml_data:
        yaml_data['adapter_conf'] = {}

    if 'target_blocks' not in yaml_data['adapter_conf']:
        yaml_data['adapter_conf']['target_blocks'] = {}

    target_blocks = yaml_data['adapter_conf']['target_blocks']

    # Add layers to target_blocks
    for layer_name, r in lora_layers:
        if layer_name not in target_blocks:
            target_blocks[layer_name] = []
        target_blocks[layer_name].append([lang_sym, task_label, r])

    return yaml_data

def main():
    # Setup logging
    setup_logging()
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Prepare paths
    model_config_path, model_file_path = prepare_paths(args)
    
    # Log the paths
    logging.info(f"Model file path: {model_file_path}")
    
    # load the model
    model = torch.load(model_file_path, map_location='cpu')

    s2b = {
            "attention": ["linear_q", "linear_k", "linear_v"],
            "attention+": ["linear_q", "linear_k", "linear_v", "linear_out"],
            "attention + feedforward": ["linear_q", "linear_k", "linear_v", "linear_out", "feed_forward", "feed_forward_macaron"],
            "all": ["linear_q", "linear_k", "linear_v", "linear_out", "channel_proj1", "channel_proj2", "feed_forward", "feed_forward_macaron", "merge_proj"]
    }
    # determine which layers should get LoRA adapters inserted
    #lora_layers = []
    #for name, param in model.items():
    #    module_name = ".".join(name.split(".")[:-1])
    #    if any([key in module_name for key in s2b[args.blocks]]) and not module_name in lora_layers:
    #        lora_layers.append(module_name)
    # determine size per LoRA adapter: default = same r for all LoRA adapters
    #lora_layers = [(key, args.r) for key in lora_layers]
    
    lora_layers = alr.process_model_layers(
            model_file='exp/ml/s2t_train_asr_owsmv31_base_raw_pretrained_bpe50000/valid.acc.ave.pth',
            interference_file='exp/ml/s2t_train_asr_owsmv31_base_raw_pretrained_bpe50000/importance_weight.ewc',
            relevance_file='exp/ml/s2t_train_asr_owsmv31_base_taskshift_raw_cgnnl_bpe50000/task_shift.pth',
            lora_layers='all',
            default_r=32,
            reduction_method='average',
            normalization_method='minmax',
            combine_method='add',
            r_multiple=1,
            threshold=0.0,
            groups=['encoder', 'decoder'],
            method='sainte-lague',
            alpha=1.0,
            no_interference=False,
            shift_values=False,
    )
    # load yaml file
    yaml_data = load_yaml(args.prev_yaml_file)
    # Update YAML configuration
    updated_yaml_data = update_yaml_config(yaml_data, lora_layers, args.lang_sym, args.task_label)
    # Save the updated YAML configuration
    save_yaml(updated_yaml_data, args.yaml_name)
    logging.info(f"Updated YAML configuration saved to: {args.yaml_name}")

    # Example logging for other arguments
    if args.prev_yaml_file:
        logging.info(f"Previous YAML file: {args.prev_yaml_file}")
    logging.info(f"Output YAML file: {args.yaml_name}")

if __name__ == "__main__":
    main()

