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
parser.add_argument("--weight", type=int, default=1, help="Weight of the old task")
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
init_ewc = torch.load("/".join(asr_train_args.init_param[0].split("/")[:-1]) + "/importance_weight.kf_last", map_location='cpu')

# load current ewc
rest_ewc = torch.load(f"{args.model}/importance_weight.ewc_ind", map_location='cpu')

# load previous ewc
rest_init_ewc = torch.load("/".join(asr_train_args.init_param[0].split("/")[:-1]) + "/importance_weight.ewc_last", map_location='cpu')


# to store new model
new_model = {}

all_values = []
weighted_mean = 0
params = 0

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
           sigma = relevance / (relevance + args.weight * interference + 1e-12)
           logging.info(f"... i={i} ... relevance = {relevance}, interference = {interference}, sigma = {sigma:.3f}")

           sigmas.append(sigma.item())

           # add gated contribution
           gated_update += sigma * S[i] * rank_update

       all_values += sigmas
       # print mean, median, max, min
       logging.info(f"mean={np.mean(sigmas):.4f}, median={np.median(sigmas):.4f}, max={np.max(sigmas):.4f}, min={np.min(sigmas):.4f}")

       # update weight in new_model
       new_model[name + ".weight"] = init_model[name + ".weight"] + gated_update


for name, param in asr_model.state_dict().items():
    if name in new_model.keys():
        continue
    delta_W = param.cpu() - init_model[name]
    rank_update_sq = delta_W.pow(2) # elementwise squared

    if not name in rest_ewc.keys():
        new_model[name] = init_model[name]
        logging.info(f"{name} ... no EWC, using old model")
        continue
    # corresponding EWC entries
    omega_old = rest_init_ewc[name]   # previous task fisher
    omega_new = rest_ewc[name]        # new task fisher

    # compute interference and relevance
    interference = torch.sum(omega_old * rank_update_sq)
    relevance    = torch.sum(omega_new * rank_update_sq)

    # compute σ(α_i)
    sigma = relevance / (relevance + args.weight * interference + 1e-12)
    logging.info(f"{name} ... relevance = {relevance}, interference = {interference}, sigma = {sigma:.3f}")

    all_values.append(sigma)

    # add gated contribution
    gated_update = sigma * delta_W
    new_model[name] = init_model[name] + gated_update


if args.weight > 1:
    suffix = str(args.weight)
else:
    suffix = ""
torch.save(new_model, f"{args.model}/kf_full_model_last{suffix}.pth")
logging.info(f"mean={np.mean(all_values):.4f}, median={np.median(all_values):.4f}, max={np.max(all_values):.4f}, min={np.min(all_values):.4f}")
logging.info(f"Done!")



