import argparse
import os
import torch
import yaml
import logging

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')

def load_model_parameters(model_path):
    return torch.load(model_path, map_location='cpu')

def compute_layer_changes(initial_params, epoch_params):
    layer_changes = {}
    for layer, initial_param in initial_params.items():
        epoch_param = epoch_params[layer]
        # Compute absolute difference and average change per parameter
        abs_diff = torch.abs(epoch_param - initial_param)
        avg_change = torch.mean(abs_diff).item()
        num_params = torch.numel(initial_param)
        layer_changes[layer] = (avg_change, num_params)
    return layer_changes

def compute_token_changes_for_layer(layer_param_diff, tokens, dim=0):
    avg_change_per_token = torch.mean(layer_param_diff, dim=dim)  # Average along D dimension
    logging.info(f"Size of avg_change_per_token: {avg_change_per_token.size(0)}")
    return dict(zip(tokens, avg_change_per_token.tolist()))

def main(model_directory):
    logging.info(f"Process task shift.. {model_directory}")
    # Load model parameters
    initial_model_path = os.path.join(model_directory, "initial_model.pth")
    epoch_model_path = os.path.join(model_directory, "1epoch.pth")
    initial_params = load_model_parameters(initial_model_path)
    epoch_params = load_model_parameters(epoch_model_path)

    # Compute and save average changes per layer
    layer_changes = compute_layer_changes(initial_params, epoch_params)
    torch.save(layer_changes, os.path.join(model_directory, "task_shift.pth"))
    logging.info("Done! Storing task shift!")

    # Load token list from YAML file
    with open(os.path.join(model_directory, "config.yaml"), 'r') as f:
        config = yaml.safe_load(f)
    tokens = config['token_list']

    logging.info("Processing token task shifts..")

    # set V
    V = len(tokens)

    logging.info(f"Found {V} tokens!")

    # Specific layers: decoder.embed and decoder.output_layer
    for layer_name in ["decoder.embed.0.weight", "decoder.output_layer.weight", "ctc.ctc_lo.weight"]:
        logging.info(f"Processing layer {layer_name} ...")
        if layer_name in initial_params:
            assert initial_params[layer_name].size(0) == V or initial_params[layer_name].size(1) == V
            # set dimension of D
            dim = 0 if initial_params[layer_name].size(1) == V else 1
            # compute absolute change
            abs_diff = torch.abs(epoch_params[layer_name] - initial_params[layer_name])
            # pair with tokens and average
            token_changes = compute_token_changes_for_layer(abs_diff, tokens, dim=dim)
            torch.save(token_changes, os.path.join(model_directory, f"task_shift_tokens{layer_name}.pth"))
            logging.info("Done! Storing token task shift!")
        else:
            logging.info("... not present in model!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute parameter changes in models.")
    parser.add_argument("model_directory", type=str, help="Directory containing model files and config")
    
    args = parser.parse_args()
    main(args.model_directory)

