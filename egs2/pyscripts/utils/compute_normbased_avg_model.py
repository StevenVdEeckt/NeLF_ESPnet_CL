import torch
import numpy as np
import argparse
import logging
from espnet2.tasks.s2t import S2TTask
from espnet2.layers.clloralib import LoRALayer

parser = argparse.ArgumentParser(description='Computing the average model...')
parser.add_argument('model',  type=str, help='Path to the model directory')
parser.add_argument('name', type=str, help="Name of the model to store")
parser.add_argument('--model_name', type=str, default="latest.pth", help="Name of the model (default = latest.pth")
parser.add_argument('--eta', type=float, default=0.5, help="Eta_a, to compute the averaging (eta_a must be a float and 0 <= eta_a <= 1)")


args = parser.parse_args()

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger('main')

# set some params
model = args.model
model_name = args.model_name
name = args.name

# printing information
logger.info("Computing the AVERAGE MODEL..")
logger.info("Loading the model from: %s" % model)
logger.info("... will use the model: %s" % model_name)

# load the model
s2t_train_config = f"{model}/config.yaml"
s2t_model_file   = f"{model}/{model_name}"
device = "cpu"

s2t_model, s2t_train_args = S2TTask.build_model_from_file(
        s2t_train_config, s2t_model_file, device
)


def get_weight(A1, A2):
    assert A1.size(0) == A2.size(0) and A1.size(1) == A2.size(1)
    # Compute row-wise squared norms
    norm_1 = torch.sum(A1 ** 2, dim=1)  # Shape: (r,)
    norm_2 = torch.sum(A2 ** 2, dim=1)  # Shape: (r,)

    # Compute eta per row
    eta = norm_2 / (norm_1 + norm_2 + 1e-8)  # Add small epsilon to avoid division by zero

    return eta

sum_eta, numel = 0.0, 0
max_eta, min_eta = 0.0, 1.0

# multiply E for new task with eta
for _name, module in s2t_model.named_modules():
    if isinstance(module, LoRALayer):
        A2, B2 = module.get_current_adapter()[:2]
        A , B  = module.get_adapter(module.lang_sym, 0)[:2]
        # 1. analyse and average A
        eta = get_weight(A, A2)
        # keep stats
        sum_eta += eta.sum().item()
        numel += eta.numel()
        if eta.max().item() > max_eta:
            max_eta = eta.max().item()
        if eta.min().item() < min_eta:
            min_eta = eta.min().item()
        logger.info(f"{_name} : min={eta.min():.3f}, mean={eta.mean():.3f}, med={eta.median():.3f}, max={eta.max():.3f}")
        A2_data = torch.diag(eta) @ A2
        A2.data.copy_(A2_data)

# print stats
logger.info(f"TOTAL: min={min_eta:.3f}, mean={sum_eta / numel:.3f}, max={max_eta:.3f}")


# get state_dict
new_state_dict = s2t_model.state_dict()

logger.info("Storing the model.. (at %s/%s" % (args.model, name))

torch.save(new_state_dict, "%s/%s" % (args.model, name))


logger.info("")
logger.info("-" * 100)

