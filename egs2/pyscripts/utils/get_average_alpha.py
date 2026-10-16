import torch
import logging
import argparse

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S %p')

parser = argparse.ArgumentParser(description='Computing the average model...')
parser.add_argument('model',  type=str, help='Path to the model directory')
args = parser.parse_args()

logging.info(f"Analysing... {args.model}")

# load model
model = torch.load(args.model + "/alphas.pth")
sigmoid = lambda x: torch.nn.functional.sigmoid(x * 10.0)

# take sum and num elements
numel = 0
alpha_sum = 0.0
max_alpha = 0.0
min_alpha = 1.0

logging.info(f"Stats per layer:")

for name, param in model.items():
    alpha_sum += param.sum().item()
    numel += param.numel()
    #a = sigmoid(param)
    a = param
    logging.info(f"... {name} : min={a.min():.3f}, med={a.median():.3f}, mean={a.mean():.3f}, max={a.max():.3f}")
    if sigmoid(param).max() > max_alpha:
        max_alpha = sigmoid(param).max()
    if sigmoid(param).min() < min_alpha:
        min_alpha = sigmoid(param).min()

avg = 1.0 * alpha_sum / numel

logging.info(f"Average alpha = {avg} (out of {numel} total rank)")
logging.info(f".... max_alpha = {max_alpha:.5f}")
logging.info(f".... min_alpha = {min_alpha:.5f}")
