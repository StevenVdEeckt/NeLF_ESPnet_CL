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
parser.add_argument("old_model", type=str, help="Directory containing the text files to process.")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# load current ewc
kf = torch.load(f"{args.model}/importance_weight.kf", map_location='cpu')
old_kf = torch.load(f"{args.old_model}/importance_weight.kf", map_location='cpu')


new_kf = {}
for name, param in kf.items():
    Q = param['Q'] + old_kf[name]['Q']
    H = param['H'] + old_kf[name]['H']
    new_kf[name] = {'Q': Q, 'H': H}

torch.save(kf, f"{args.model}/importance_weight.kf_single_old")
torch.save(new_kf, f"{args.model}/importance_weight.kf")

