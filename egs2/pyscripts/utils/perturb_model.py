import argparse
import torch
import torch.nn as nn
import os

def perturb_layer(layer, perturbation_factor):
    """
    Apply perturbation to a given layer.
    """
    with torch.no_grad():
        if hasattr(layer, 'weight') and layer.weight is not None:
            layer.weight += torch.randn(layer.weight.size()) * perturbation_factor
        if hasattr(layer, 'bias') and layer.bias is not None:
            layer.bias += torch.randn(layer.bias.size()) * perturbation_factor


def main():
    parser = argparse.ArgumentParser(description='Perturb layers in a PyTorch model.')
    parser.add_argument('model', type=str, help='Path to the PyTorch model file.')
    parser.add_argument('layers', type=str, help='Comma separated list of layers to perturb.')
    parser.add_argument('perturbation_factor', type=float, help='Perturbation factor to apply to the layers.')
    parser.add_argument('new_model_name', type=str, help='Name for the new perturbed model file.')
    
    args = parser.parse_args()
    
    model_path = args.model
    layers_to_perturb = args.layers.split(',')
    perturbation_factor = args.perturbation_factor
    new_model_name = args.new_model_name

    # Load the model
    model = torch.load(model_path, map_location='cpu')

    # Apply perturbation to specified layers
    for name, layer in model.items():
        if any(name.startswith(layer_name) for layer_name in layers_to_perturb):
            if not 'norm' in name:
                perturb_layer(layer, perturbation_factor)
                print(f'Perturbed layer: {name}')
            else:
                print(f'Skipped normalization layer: {name}')
    
    # Save the new model
    new_model_path = os.path.join(os.path.dirname(model_path), new_model_name)
    torch.save(model, new_model_path)
    print(f'New model saved as: {new_model_path}')

if __name__ == '__main__':
    main()

