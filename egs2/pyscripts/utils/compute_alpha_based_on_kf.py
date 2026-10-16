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

# load previous model
#init_model = torch.load(asr_train_args.init_param[0], map_location='cpu')
init_model = torch.load(f"{args.model}/initial_model.pth", map_location='cpu')

# load current ewc
ewc = torch.load(f"{args.model}/importance_weight.kf_ind", map_location='cpu')

# load previous ewc
init_ewc = torch.load("/".join(asr_train_args.init_param[0].split("/")[:-1]) + "/importance_weight.kf", map_location='cpu')

# to store new model
new_model = {name: 0.5*(param.float() + init_model[name].float()) for name, param in asr_model.state_dict().items()}

#def loss(update, H_list, Q_list):
#    loss = 0.0
#    for H, Q in zip(H_list, Q_list):
        #logging.info(f"Sizes: update = {update.size()}, H = {H.size()}, Q = {Q.size()}")
#        HWQ = torch.matmul(H, torch.matmul(update.transpose(0, 1), Q))
#        loss += torch.dot((update).view(-1), HWQ.view(-1))
#    return loss

def loss(update, H_list, Q_list):
    loss = 0.0
    for H, Q in zip(H_list, Q_list):
        loss += torch.trace(Q @ update @ H @ update.T)
    return loss



for name, module in asr_model.named_modules():
    if isinstance(module, torch.nn.Linear):
       # compute difference
       delta_W = module.weight.data.cpu() - init_model[name + ".weight"]

       logging.info(f"Processing layer... {name}")
       # compute SVD
       U,S,Vh = torch.linalg.svd(delta_W, full_matrices=False)
       # rebuild update with gated singular values
       gated_update = torch.zeros_like(delta_W)

       sigmas = []

       for i in range(S.size(0)):
           ui = U[:, i].unsqueeze(1)           # (out, 1)
           vi = Vh[i, :].unsqueeze(0)          # (1, in)
           rank_update = (ui @ vi)             # outer product, shape (out, in)

           # compute interference and relevance
           interference = loss(rank_update, init_ewc[name]['Q'], init_ewc[name]['H']) 
           relevance    = loss(-rank_update, ewc[name]['Q'], ewc[name]['H'])

           # compute σ(α_i)
           sigma = relevance / (relevance + interference + 1e-12)
           logging.info(f"... i={i} ... relevance = {relevance}, interference = {interference}, sigma = {sigma:.3f}")

           sigmas.append(sigma.item())

           # add gated contribution
           gated_update += sigma * S[i] * rank_update

       # print mean, median, max, min
       logging.info(f"mean={np.mean(sigmas):.4f}, median={np.median(sigmas):.4f}, max={np.max(sigmas):.4f}, min={np.min(sigmas):.4f}")

       # update weight in new_model
       new_model[name + ".weight"] = init_model[name + ".weight"] + gated_update
       # biases: average old and new (as in original SVR)
       if module.bias is not None:
           new_model[name + ".bias"] = 0.5 * (init_model[name + ".bias"] + module.bias.data.cpu())

torch.save(new_model, f"{args.model}/kf_model.pth")
logging.info(f"Done!")



