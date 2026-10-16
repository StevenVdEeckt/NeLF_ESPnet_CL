import torch
import argparse
import logging


def get_layer(layer_name):
    return ".".join(layer_name.split(".")[:-2])

def analyse_increlora(model, zero_threshold=1e-5, almost_zero_threshold=0.01):
    # step 1. load all lora_E
    adapters = {k: v for k, v in model.items() if 'lora_E' in k}
    layer_overview = {}
    # 2. group lora_E per layer
    for k, v in adapters.items():
        layer = get_layer(k)
        if not layer in layer_overview.keys():
            layer_overview[layer] = []
        layer_overview[layer].append(v.item())
    # 3. analyse per layer
    assert almost_zero_threshold >= zero_threshold
    max_len = max([len(k) for k in layer_overview.keys()])
    for layer, values in layer_overview.items():
        zeros = len([v for v in values if v < zero_threshold])
        almost_zeros = len([v for v in values if zero_threshold <= v < almost_zero_threshold])
        total = len(values)
        rest = total - zeros - almost_zeros
        if rest > 0:
            avg_nonzero_value = sum([v for v in values if v >= almost_zero_threshold]) / rest
        else:
            avg_nonzero_value = 0
        space = " " * (max_len - len(layer))
        pvalues = [0.0 if p < almost_zero_threshold else round(p, 2) for p in values]
        logging.info(f"{layer}{space} --> total: {total}, zeros: {zeros}, almost_zeros: {almost_zeros}, nonzeros: {rest} ({round(100.0*rest / total, 2)}%), values: {pvalues}")

def main():
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')
    parser = argparse.ArgumentParser(description='Merging datasets')
    parser.add_argument('model', type=str, help='Path to the model - either to the model directory or the epoch file.')

    args = parser.parse_args()

    model = args.model
    if '.pth' in model:
        logging.info(f"Analysing model: {model}")
        m = torch.load(model, map_location='cpu')
    else:
        logging.info(f"Analysing model: {model}/latest.pth")
        m = torch.load(f"{model}/latest.pth", map_location='cpu')

    analyse_increlora(m)

# RUN MAIN FILE
if __name__ == "__main__":
    main()
