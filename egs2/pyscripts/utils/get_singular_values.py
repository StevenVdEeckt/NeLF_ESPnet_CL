from espnet2.tasks.asr import ASRTask
import torch
import logging
from torch import nn
from decimal import Decimal
import os
import argparse


parser = argparse.ArgumentParser(description="Process text files to add punctuation, capitalize sentences, and handle special tokens.")
parser.add_argument("model", type=str, help="Directory containing the text files to process.")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

asr_train_config = f"{args.model}/config.yaml"
asr_model_file   = f"{args.model}/valid.acc.ave.pth"
device = "cpu"

asr_model, asr_train_args = ASRTask.build_model_from_file(
        asr_train_config, asr_model_file, device
)

init_model = torch.load(asr_train_args.init_param[0], map_location='cpu')

singular_values = {}

for name, module in asr_model.named_modules():
    if isinstance(module, torch.nn.Linear):
       weight = module.weight.data.cpu() - init_model[name + ".weight"]
       _, S, _ = torch.linalg.svd(weight, full_matrices=False)
       singular_values[name] = S.detach()

torch.save(singular_values, f"{args.model}/singular_values.pth")
logging.info("Done loading!")

