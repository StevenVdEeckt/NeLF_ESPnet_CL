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

def compute_scores(interference, relevance, combine_method='multiply'):
    """Combine interference and relevance scores."""
    if combine_method == 'multiply':
        return np.maximum(interference, 0) * np.maximum(relevance, 0)
    elif combine_method == 'add':
        return interference + relevance
    else:
        raise ValueError("Unknown combine method: {}".format(combine_method))

def transform(normalized_interference, normalized_relevance, alpha=1.0, epsilon=1e-5, shift=False):
    """
    Transform normalized interference and relevance scores into a combined score.

    Args:
    - normalized_interference (np.array): Normalized interference scores.
    - normalized_relevance (np.array): Normalized relevance scores.
    - alpha (float): Hyper-parameter to control the distribution.
    - epsilon (float): Small constant to avoid zero values.

    Returns:
    - combined_scores (np.array): Transformed and combined scores.
    """
    # Ensure inputs are numpy arrays
    normalized_interference = np.array(normalized_interference)
    normalized_relevance = np.array(normalized_relevance)

    # Calculate the constant c to shift the scores to be non-negative
    # Calculate the constants to shift the scores to be non-negative
    if shift:
        shift_interference = abs(np.min(normalized_interference)) + epsilon
        shift_relevance = abs(np.min(normalized_relevance)) + epsilon
    else:
        shift_interference = 0.0
        shift_relevance = 0.0

    # Shift the scores
    shifted_interference = normalized_interference + shift_interference
    shifted_relevance = normalized_relevance + shift_relevance

    # Apply the power transformation with alpha
    transformed_interference = shifted_interference ** alpha
    transformed_relevance = shifted_relevance ** alpha
    return transformed_interference, transformed_relevance


def assign_r_values(layers, scores, total_r, r_multiple, threshold=0.0, method='hare'):
    if method == 'hare':
        return assign_r_values_hare(layers, scores, total_r, r_multiple, threshold)
    elif method == 'dhont': 
        return assign_r_values_dhont(layers, scores, total_r, r_multiple, threshold)
    elif method in ["sainte-lague", "sl", "sainte_lague"]:
        return assign_r_values_sainte_lague(layers, scores, total_r, r_multiple, threshold)
    else:
        raise NotImplementedError("Method %s not implemented" % method)

def assign_r_values_dhont(layers, scores, total_r, r_multiple, threshold=0.0):
    """Allocate r values using the D'Hondt method."""
    scores = np.array(scores)
    if threshold >= 0.0:
        scores[scores < threshold] = 0.0
    
    total_seats = total_r // r_multiple
    seat_allocation = np.zeros(len(scores))
    divisors = np.ones(len(scores))

    for _ in range(total_seats):
        quotients = scores / divisors
        max_index = np.argmax(quotients)
        seat_allocation[max_index] += 1
        divisors[max_index] += 1

    r_values = seat_allocation * r_multiple
    return r_values.astype(int)

def assign_r_values_hare(layers, scores, total_r, r_multiple, threshold=0.0):
    """Allocate r values using the Hare Quota method."""
    scores = np.array(scores)
    if threshold >= 0.0:
        scores[scores < threshold] = 0.0

    total_seats = total_r // r_multiple
    quota = np.sum(scores) / total_seats

    # Initial allocation based on quota
    seat_allocation = np.floor(scores / quota).astype(int)
    remaining_seats = total_seats - np.sum(seat_allocation)

    # Calculate remainders
    remainders = scores - seat_allocation * quota

    # Distribute remaining seats based on largest remainders
    for _ in range(remaining_seats):
        max_index = np.argmax(remainders)
        seat_allocation[max_index] += 1
        remainders[max_index] = -1  # Mark this index as processed

    r_values = seat_allocation * r_multiple
    return r_values.astype(int)

def assign_r_values_sainte_lague(layers, scores, total_r, r_multiple, threshold=0.0):
    """Allocate r values using the Sainte-Laguë method."""
    scores = np.array(scores)
    if threshold >= 0.0:
        scores[scores < threshold] = 0.0

    total_seats = total_r // r_multiple
    seat_allocation = np.zeros(len(scores))
    divisors = np.ones(len(scores))

    for _ in range(total_seats):
        quotients = scores / (2 * seat_allocation + 1)
        max_index = np.argmax(quotients)
        seat_allocation[max_index] += 1

    r_values = seat_allocation * r_multiple
    return r_values.astype(int)

def generate_table(layers, interference, relevance, scores, r_values):
    """Generate and print a table of layer information."""
    print("")
    headers = ["Layer Name", "Interference (I)", "Relevance (R)", "Score", "r Value"]
    rows = []
    for layer, I, R, score, r in zip(layers, interference, relevance, scores, r_values):
        rows.append([layer, "%.2f" % I, "%.2f" % R, "%.2f" % score, r])
    
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

def generate_table_per_block(selected_layers, normalized_interference, normalized_relevance, scores, r_values, get_block=None):
    # Generate and print table per encoder and decoder layer
    per_block = {}
    def _get_block(layer):
        return ".".join(layer.split(".")[:3])
    if get_block is None:
        get_block = _get_block
    assert len(selected_layers) == len(normalized_interference) == len(normalized_relevance) == len(scores) == len(r_values)
    for layer, interference, relevance, score, r_value in zip(selected_layers, normalized_interference, normalized_relevance, scores, r_values):
        block = get_block(layer)
        if block in per_block.keys():
            per_block[block][0].append(interference)
            per_block[block][1].append(relevance)
            per_block[block][2].append(score)
            per_block[block][3].append(r_value)
        else:
            per_block[block] = ([interference], [relevance], [score], [r_value])
    selected_layers, normalized_interference, normalized_relevance, scores, r_values = [], [], [], [], []
    for block, values in per_block.items():
        selected_layers.append(block)
        normalized_interference.append(sum(values[0]))
        normalized_relevance.append(sum(values[1]))
        scores.append(sum(values[2]))
        r_values.append(sum(values[3]))
    generate_table(selected_layers, normalized_interference, normalized_relevance, scores, r_values)

def process_model_layers(
        model_file, 
        interference_file, 
        relevance_file, 
        lora_layers, 
        default_r, 
        reduction_method, 
        normalization_method, 
        combine_method, 
        r_multiple, 
        threshold,
        groups,
        method,
        alpha,
        no_interference=False,
        shift_values=False,
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
        # transform 
        interference, relevance = transform(interference, relevance, alpha, epsilon=1e-5, shift=shift_values)
        # Compute combined scores
        scores = compute_scores(interference, relevance, combine_method)
        # Assign r values
        total_r = default_r * len(layers)
        r_values = assign_r_values(layers, scores, total_r, r_multiple, threshold, method)
        values.append(list(scores))
        # combine
        selected_layers += list(layers)
        tot_interference += list(interference)
        tot_relevance += list(relevance)
        tot_scores += list(scores)
        tot_r_values += list(r_values)
        assert len(selected_layers) == len(tot_interference) == len(tot_relevance) == len(tot_scores) == len(tot_r_values)

    lora_layers = [(".".join(key.split(".")[:-1]), int(r)) for (key, r) in zip(selected_layers, tot_r_values) if r > 0]

    # Generate and print table
    generate_table(selected_layers, tot_interference, tot_relevance, tot_scores, tot_r_values)

    # Generate and print table per encoder and decoder layer
    generate_table_per_block(selected_layers, tot_interference, tot_relevance, tot_scores, tot_r_values)

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
    generate_table_per_block(selected_layers, tot_interference, tot_relevance, tot_scores, tot_r_values, get_block)
    return lora_layers

# Example usage of process_model_layers function
if __name__ == "__main__":
    setup_logging()

    # Example values, replace with actual paths and parameters
    model_file = 'exp/ml/s2t_train_asr_owsmv31_base_raw_pretrained_bpe50000/valid.acc.ave.pth'
    interference_file = 'exp/ml/s2t_train_asr_owsmv31_base_raw_pretrained_bpe50000/importance_weight.ewc'
    relevance_file = 'exp/ml/s2t_train_asr_owsmv31_base_taskshift_raw_cgnnl_bpe50000/task_shift.pth'
    no_interference = False
    lora_layers = "all"
    default_r = 8
    reduction_method = 'average'
    normalization_method = 'minmax'
    combine_method = 'add'
    r_multiple = 1
    method = 'sainte-lague'
    threshold = 0.0
    shift_values = False
    alpha = 1.0
    groups = ["encoder", "decoder"]

    process_model_layers(
            model_file, 
            interference_file, 
            relevance_file, 
            lora_layers, 
            default_r, 
            reduction_method, 
            normalization_method, 
            combine_method, 
            r_multiple, 
            threshold,
            groups,
            method,
            alpha,
            no_interference,
            shift_values,
    )

