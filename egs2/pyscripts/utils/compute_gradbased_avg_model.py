import os
import torch
import argparse
import logging

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S %p')

def parse_args():
    parser = argparse.ArgumentParser(description='Average models based on gradients.')
    parser.add_argument('--model', required=True, help='Directory of the adapted model')
    parser.add_argument('--old_model', required=True, help='Directory of the old model')
    parser.add_argument('--model_name', default='valid.acc.ave.pth', help='Name of the model file (default: "valid.acc.ave.pth")')
    parser.add_argument('--old_gradient', default='importance_weight.gradient.old')
    parser.add_argument('--new_gradient', default='importance_weight.gradient.new')
    parser.add_argument('--name', required=True, help='Name of the averaged model to be computed')
    return parser.parse_args()

def check_files(model_dir, model_name, gradient_name=[]):
    model_path = os.path.join(model_dir, model_name)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    gradient_paths = []
    for grad_name in gradient_name:
        gradient_path = os.path.join(model_dir, grad_name)
        if not os.path.isfile(gradient_path):
            raise FileNotFoundError(f"Gradient file not found: {gradient_path}")
        gradient_paths.append(gradient_path)
    return model_path, gradient_paths

def load_model_and_gradients(model_path, gradient_path):
    model = torch.load(model_path)
    gradients = torch.load(gradient_path)
    return model, gradients

def compute_average_weights_v1(grad_dict_old, grad_dict_new, excluded_layers):
    # Collect all gradients into a single large tensor
    all_grads_old = torch.cat([v.view(-1) for k, v in grad_dict_old.items() if not k in excluded_layers])
    all_grads_new = torch.cat([v.view(-1) for k, v in grad_dict_new.items() if not k in excluded_layers])
    
    # Compute cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(all_grads_old, all_grads_new, dim=0)
    
    # Map cosine similarity to weight
    weight = map_cosine_similarity(cos_sim.item())
    
    # Create weight dictionary
    weight_dict = {layer: weight for layer in grad_dict_old.keys()}
    return weight_dict

def compute_average_weights_v2(grad_dict_old, grad_dict_new, excluded_layers):
    weight_dict = {}
    for layer in grad_dict_old.keys():
        grad_old = grad_dict_old[layer].view(-1)
        grad_new = grad_dict_new[layer].view(-1)
        
        # Compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(grad_old, grad_new, dim=0)
        
        # Map cosine similarity to weight
        weight = map_cosine_similarity(cos_sim.item())
        logging.info("%s --> cos_sim = %.2f => w = %.2f" % (layer, cos_sim, weight)) 
        weight_dict[layer] = weight
    return weight_dict

def compute_average_weights_groupbased(grad_dict_old, grad_dict_new, excluded_layers, groups):
    weight_dict = {}
    if groups:
        layer_groups = {group: {} for group in groups}
        layer_groups['other'] = {}

        for layer_name in grad_dict_old.keys():
            matched = False
            for group in groups:
                if layer_name.startswith(group):
                    layer_groups[group][layer_name] = grad_dict_old[layer_name]
                    matched = True
                    break
            if not matched:
                layer_groups['other'][layer_name] = grad_dict_old[layer_name]
    else:
        # If no groups are provided, treat all layers as a single group
        layer_groups = {'all': grad_dict_old}

    # Apply OGD to each group separately
    for group, layers in layer_groups.items():
        if not layers:
            continue
            
        # Collect gradients for the current group
        g_new = []
        g_old = []
        for name, param in grad_dict_new.items():
            if name in layers:
                g_new.append(param.view(-1))
                g_old.append(layers[name].view(-1))
            
        # Convert lists to tensors
        g_new = torch.cat(g_new)
        g_old = torch.cat(g_old)
            
        # Compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(g_old, g_new, dim=0)

        # Map cosine similarity to weight
        weight = map_cosine_similarity(cos_sim.item())
        logging.info("%s --> cos_sim = %.2f => w = %.2f" % (group, cos_sim, weight))
        for layer in layers:
            weight_dict[layer] = weight
    print(f"weight_dict = {weight_dict}")
    return weight_dict


def map_cosine_similarity(cos_sim):
    # Example implementation of the mapping function
    return (1 + cos_sim) / 2

def compute_new_model(old_model, adapted_model, weight_dict):
    new_model = {}
    for layer in old_model.keys():
        if layer in weight_dict.keys():
            new_model[layer] = (1 - weight_dict[layer]) * old_model[layer] + weight_dict[layer] * adapted_model[layer]
        else:
            new_model[layer] = adapted_model[layer]
    return new_model

def main():
    args = parse_args()

    # Check if files exist
    old_model_path, gradient_paths = check_files(args.old_model, args.model_name, [args.old_gradient, args.new_gradient])
    adapted_model_path, _ = check_files(args.model, args.model_name, [])

    groups = None #["encoder.encoders.0.adapter1",
             #"encoder.encoders.0.adapter2",
             #"encoder.encoders.1.adapter1",
             #"encoder.encoders.1.adapter2",
             #"encoder.encoders.2.adapter1",
             #"encoder.encoders.2.adapter2",
             #"encoder.encoders.3.adapter1",
             #"encoder.encoders.3.adapter2",
             #"encoder.encoders.4.adapter1",
             #"encoder.encoders.4.adapter2",
             #"encoder.encoders.5.adapter1",
             #"encoder.encoders.5.adapter2"]

    # Load models and gradients
    old_model, old_gradients = load_model_and_gradients(old_model_path, gradient_paths[0])
    adapted_model, adapted_gradients = load_model_and_gradients(adapted_model_path, gradient_paths[1])

    excluded_layers = [k for k in old_gradients.keys()]

    # Compute average weights (choose implementation here)
    weight_dict = compute_average_weights_groupbased(old_gradients, adapted_gradients, excluded_layers, groups)  # or compute_average_weights_v1

    # Compute new model
    new_model = compute_new_model(old_model, adapted_model, weight_dict)

    # Save new model
    new_model_path = os.path.join(args.model, args.name)
    torch.save(new_model, new_model_path)
    print(f"Averaged model saved at {new_model_path}")

if __name__ == '__main__':
    main()

