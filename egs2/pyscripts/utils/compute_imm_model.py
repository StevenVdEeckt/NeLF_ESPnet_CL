from espnet2.tasks.asr import ASRTask
import torch
import logging
from torch import nn
from decimal import Decimal
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Process text files to add punctuation, capitalize sentences, and handle special tokens.")
parser.add_argument("model", type=str, help="Directory containing the text files to process.")
parser.add_argument("--alpha", default=1.0, type=float, help="Directory containing the text files to process.")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# load current model
asr_train_config = f"{args.model}/config.yaml"
asr_model_file   = f"{args.model}/valid.acc.ave.pth"
device = "cpu"

asr_model, asr_train_args = ASRTask.build_model_from_file(
        asr_train_config, asr_model_file, device
)

# load previous model
init_model = torch.load(f"{args.model}/initial_model.pth", map_location='cpu')
#init_model = torch.load(asr_train_args.init_param[0], map_location='cpu')

# load current ewc
ewc = torch.load(f"{args.model}/importance_weight.ewc_ind", map_location='cpu')

# load previous ewc
init_ewc = torch.load("/".join(asr_train_args.init_param[0].split("/")[:-1]) + "/importance_weight.ewc", map_location='cpu')

all_values = []

new_model = {}

# to store new model
for name, param in asr_model.state_dict().items():
    if not name in ewc.keys():
        new_model[name] = 0.5 * (param.float() + init_model[name].float())
    else:
        logging.info(f"Processing layer... {name}")
        eta = args.alpha * ewc[name] / (args.alpha * ewc[name] + init_ewc[name] + 1e-12) 
        all_values += eta.view(-1).tolist()
        logging.info(f"mean={torch.mean(eta):.4f}, median={torch.median(eta):.4f}, max={torch.max(eta):.4f}, min={torch.min(eta):.4f}")
        new_param = (1 - eta) * init_model[name] + eta * param
        new_model[name] = new_param

if args.alpha > 1.0:
    suffix=f"{args.alpha}"
else:
    suffix=""

torch.save(new_model, f"{args.model}/imm_model{suffix}.pth")
logging.info(f"mean={np.mean(all_values):.4f}, median={np.median(all_values):.4f}, max={np.max(all_values):.4f}, min={np.min(all_values):.4f}")
logging.info(f"Done!")



