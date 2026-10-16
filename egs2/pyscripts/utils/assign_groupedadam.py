import torch
import logging
import numpy as np
from pathlib import Path
from scipy.stats import zscore
from sklearn.preprocessing import minmax_scale, robust_scale

import torch.nn.functional as F

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

def reduce_weights(weights, method='average'):
    """Reduce weights tensor to a scalar using the specified method."""
    if method == 'average':
        return torch.mean(weights).item()
    elif method == 'max':
        return torch.max(weights).item()
    else:
        raise ValueError("Unknown reduction method: {}".format(method))

def normalize_weights(weights, method='minmax'):
    """Normalize weights using the specified method."""
    if method == 'minmax':
        return minmax_scale(weights)
    elif method == 'zscore':
        return zscore(weights)
    elif method == 'iqr':
        return robust_scale(weights, with_centering=False)
    elif method == "none":
        return weights
    else:
        raise ValueError("Unknown normalization method: {}".format(method))

def get_lora_layers(blocks, model):
    s2b = {
            "attention": ["linear_q", "linear_k", "linear_v"],
            "attention+": ["linear_q", "linear_k", "linear_v", "linear_out"],
            "attention + feedforward": ["linear_q", "linear_k", "linear_v", "linear_out", "feed_forward", "feed_forward_macaron"],
            "all": ["linear_q", "linear_k", "linear_v", "linear_out", "channel_proj1", "channel_proj2", "feed_forward", "feed_forward_macaron","merge_proj"]
    }
    # determine which layers should get LoRA adapters inserted
    lora_layers = []
    for name, param in model.items():
        if 'bias' in name:
            continue
        module_name = name
        if any([key in module_name for key in s2b[blocks]]) and not module_name in lora_layers:
            lora_layers.append(module_name)
    return lora_layers

def compute_scores(interference, relevance):
    """Combine interference and relevance scores."""
    x = relevance - interference 
    x = np.maximum(x, 0)
    #x normalize_weights(x, method='minmax')
    return x

def generate_table(layers, interference, relevance, scores):
    """Generate and print a table of layer information."""
    print("")
    headers = ["Layer Name", "Interference (I)", "Relevance (R)", "Score"]
    rows = []
    for layer, I, R, score in zip(layers, interference, relevance, scores):
        rows.append([layer, "%.2f" % I, "%.2f" % R, "%.2f" % score])
    
    rows = sorted(rows, key=lambda x: x[0])  # Sort by layer name

    # Determine column widths
    col_widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]
    col_widths = [max(len(headers[i]), col_widths[i]) for i in range(len(headers))]

    # Print header
    header_row = " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers)))
    print(header_row)
    print("-" * len(header_row))

    # Print rows
    for row in rows:
        print(" | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers))))

def generate_table_per_block(selected_layers, normalized_interference, normalized_relevance, scores, get_block=None):
    # Generate and print table per encoder and decoder layer
    per_block = {}
    def _get_block(layer):
        return ".".join(layer.split(".")[:3])
    if get_block is None:
        get_block = _get_block
    assert len(selected_layers) == len(normalized_interference) == len(normalized_relevance) == len(scores)
    for layer, interference, relevance, score in zip(selected_layers, normalized_interference, normalized_relevance, scores):
        block = get_block(layer)
        if block in per_block.keys():
            per_block[block][0].append(interference)
            per_block[block][1].append(relevance)
            per_block[block][2].append(score)
        else:
            per_block[block] = ([interference], [relevance], [score])
    selected_layers, normalized_interference, normalized_relevance, scores = [], [], [], []
    for block, values in per_block.items():
        selected_layers.append(block)
        normalized_interference.append(sum(values[0]))
        normalized_relevance.append(sum(values[1]))
        scores.append(sum(values[2]))
    generate_table(selected_layers, normalized_interference, normalized_relevance, scores)

def process_model_layers(
        model_file, 
        interference_file, 
        relevance_file, 
        lora_layers, 
        reduction_method, 
        normalization_method, 
        groups,
        no_interference=False,
        mean_lr=1.0,
    ):
    """Main function to process model layers and assign LoRA adapter r values."""
    # Load model and weight files
    model_state = torch.load(model_file, map_location='cpu')
    interference_state = torch.load(interference_file, map_location='cpu')
    relevance_state = torch.load(relevance_file, map_location='cpu')
    if no_interference:
        interference_state = {name: torch.zeros_like(param) for name, param in interference_state.items()}

    # lora layers
    lora_layers = get_lora_layers(lora_layers, model_state)

    # Select layers for LoRA insertion and reduce weights
    selected_layers = lora_layers
    interference_values = [reduce_weights(interference_state[layer], reduction_method) for layer in selected_layers]
    relevance_values = [relevance_state[layer][0] for layer in selected_layers]

    assert len(selected_layers) == len(interference_values) == len(relevance_values)
    # layer_groups
    layer_groups = {}
    for name, inf, rel in zip(selected_layers, interference_values, relevance_values):
        groupkey = "rest"
        for group in groups:
            if group in name:
                groupkey = group
        if groupkey in layer_groups.keys():
            layer_groups[groupkey][0].append(name)
            layer_groups[groupkey][1].append(inf)
            layer_groups[groupkey][2].append(rel)
        else:
            layer_groups[groupkey] = [[name], [inf], [rel], []]

    selected_layers, tot_interference, tot_relevance, tot_scores, tot_r_values = [], [], [], [], []
    # per layer group
    for group, values in layer_groups.items():
        interference, relevance, layers = values[1], values[2], values[0]
        assert len(interference) == len(relevance) == len(layers)
        # normalize
        interference = normalize_weights(interference, normalization_method)
        relevance = normalize_weights(relevance, normalization_method)
        # Compute combined scores
        scores = compute_scores(interference, relevance)
        values.append(list(scores))
        # combine
        selected_layers += list(layers)
        tot_interference += list(interference)
        tot_relevance += list(relevance)
        tot_scores += list(scores)
        assert len(selected_layers) == len(tot_interference) == len(tot_relevance) == len(tot_scores)

    # mean_lr: multiply lr from lora_layers by alpha such that lora_layers.mean()=mean_lr
    mean_lora_lr = sum(tot_scores) / len(tot_scores)
    mean_lr_multiplier = mean_lr / mean_lora_lr
    logging.info(f"Learning rates will be multiplied by {mean_lr_multiplier} (to achieve desired mean LR {mean_lr})")

    _lora_layers = [(key, mean_lr_multiplier * r) for (key, r) in zip(selected_layers, tot_scores)] 

    # Generate and print table
    generate_table(selected_layers, tot_interference, tot_relevance, tot_scores)

    # Generate and print table per encoder and decoder layer
    generate_table_per_block(selected_layers, tot_interference, tot_relevance, tot_scores)

    # Generate and print table per layer type
    layer_types = [
            "encoder.attn.linear_q",
            "encoder.attn.linear_v",
            "encoder.attn.linear_k",
            "encoder.attn.linear_out",
            "encoder.cgmlp.channel_proj1",
            "encoder.cgmlp.channel_proj2",
            "encoder.feed_forward.w_1",
            "encoder.feed_forward.w_2",
            "encoder.feed_forward_macaron.w_1",
            "encoder.feed_forward_macaron.w_2",
            "encoder.merge_proj",
            "decoder.self_attn.linear_q",
            "decoder.self_attn.linear_k",
            "decoder.self_attn.linear_v",
            "decoder.self_attn.linear_out",
            "decoder.src_attn.linear_k",
            "decoder.src_attn.linear_q",
            "decoder.src_attn.linear_v",
            "decoder.src_attn.linear_out",
            "decoder.feed_forward.w_1",
            "decoder.feed_forward.w_2",
    ]
    def contains_keywords(layer_name, keywords_string):
        """Check if all keywords from the keywords_string appear in the layer_name."""
        keywords = keywords_string.split('.')
        return all(keyword in layer_name for keyword in keywords)
    def get_block(layer):
        for layer_type in layer_types:
            if contains_keywords(layer, layer_type):
                return layer_type
        raise Exception("Layer types are not well defined; %s does not belong to any layer_type" % layer)
    generate_table_per_block(selected_layers, tot_interference, tot_relevance, tot_scores, get_block) 
    return _lora_layers

# Example usage of process_model_layers function
if __name__ == "__main__":
    setup_logging()

    # Example values, replace with actual paths and parameters
    model_file = 'exp/ml/s2t_train_asr_owsmv31_base_raw_pretrained_bpe50000/valid.acc.ave.pth'
    interference_file = 'exp/ml/s2t_train_asr_owsmv31_base_raw_pretrained_bpe50000/importance_weight.ewc'
    relevance_file = 'exp/ml/s2t_train_asr_owsmv31_base_taskshift_raw_cgnnl_bpe50000/task_shift.pth'
    no_interference = False
    lora_layers = "all"
    reduction_method = 'average'
    normalization_method = 'minmax'
    groups = ["encoder", "decoder"]

    process_model_layers(
            model_file, 
            interference_file, 
            relevance_file, 
            lora_layers, 
            reduction_method, 
            normalization_method, 
            groups,
            no_interference,
    )

