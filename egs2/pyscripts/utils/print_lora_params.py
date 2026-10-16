import argparse
import torch
import logging
import os


def format_number(num):
    """Format numbers into human-readable form with units."""
    for unit in ['', 'K', 'M', 'B']:
        if abs(num) < 1000.0:
            return f"{num:.2f}{unit}"
        num /= 1000.0
    return f"{num:.2f}B"


def calculate_parameters(state_dict):
    """Calculate total, adapter, and shared parameters."""
    total_params = 0
    adapter_params = 0
    total_rank = 0

    for name, param in state_dict.items():
        if not torch.is_tensor(param):
            continue
        num_params = param.numel()
        total_params += num_params
        if 'adapter' in name.lower() and not 'ranknum' in name.lower():
            adapter_params += num_params
            #logging.info(f"Adding {num_params} parameters for adapters for layer {name} --> total = {adapter_params}")
        elif 'adapter' in name.lower() and 'ranknum' in name.lower():
            total_rank += param.item()
    shared_params = total_params - adapter_params
    return total_params, adapter_params, shared_params, total_rank


def main():
    parser = argparse.ArgumentParser(description="Load model and compute parameter statistics.")
    parser.add_argument('model_dir', type=str, help="Directory where the model file is located.")
    parser.add_argument('--model_file', type=str, default='latest.pth', help="Model file name (default: latest.pth).")
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, args.model_file)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file {model_path} does not exist.")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f"Loading model from: {args.model_dir}/{args.model_file}")

    # Load model state_dict
    state_dict = torch.load(model_path, map_location='cpu')

    # Calculate parameters
    total_params, adapter_params, shared_params, total_rank = calculate_parameters(state_dict)

    # Compute percentages
    adapter_percentage = (adapter_params / total_params) * 100 if total_params > 0 else 0
    shared_percentage = (shared_params / total_params) * 100 if total_params > 0 else 0

    # Print results
    logging.info(f"  Total parameters: {format_number(total_params)}")
    logging.info(f" Shared parameters: {format_number(shared_params)} ({shared_percentage:.3f}%)")
    logging.info(f"Adapter parameters: {format_number(adapter_params)} ({adapter_percentage:.3f}%)")
    logging.info(f"   Total LoRA rank: {int(total_rank)}")

if __name__ == "__main__":
    main()

