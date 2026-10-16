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
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# load current model
asr_train_config = f"{args.model}/config.yaml"
asr_model_file   = f"{args.model}/valid.acc.ave.pth"
device = "cpu"

asr_model, asr_train_args = ASRTask.build_model_from_file(
        asr_train_config, asr_model_file, device
)

# load current ewc
ewc = torch.load(f"{args.model}/importance_weight.ewc_ind", map_location='cpu')

# load previous ewc
init_ewc = torch.load("/".join(asr_train_args.init_param[0].split("/")[:-1]) + "/importance_weight.ewc", map_location='cpu')

logging.info(f"Merging EWC's...")
logging.info("... " + "/".join(asr_train_args.init_param[0].split("/")[:-1]) + "/importance_weight.ewc")
logging.info(f"... {args.model}/importance_weight.ewc_ind")

# new ewc
ewc = {name: param + init_ewc[name] for name, param in ewc.items()}

logging.info(f" storing at: {args.model}/importance_weight.ewc")

torch.save(ewc, f"{args.model}/importance_weight.ewc")
logging.info(f"Done!")
