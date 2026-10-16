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
parser.add_argument('--eta_ort', type=float, default=0.5, help="Eta_a, to compute the averaging (eta_a must be a float and 0 <= eta_a <= 1)")
parser.add_argument('--eta_par', type=float, default=0.5, help="Eta_b, to compute the averaging (eta_b must be a float and 0 <= eta_b <= 1)")


args = parser.parse_args()

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger('main')

# set some params
model = args.model
model_name = args.model_name
eta_a, eta_b = args.eta_par, args.eta_ort
assert 0 <= eta_a <= 1 and 0 <= eta_b <= 1
name = args.name

# printing information
logger.info("Computing the AVERAGE MODEL..")
logger.info("Loading the model from: %s" % model)
logger.info("... will use the model: %s" % model_name)
logger.info("... eta will be: (A=%.3f, B=%.3f)" % (eta_a, eta_b))

# load the model
s2t_train_config = f"{model}/config.yaml"
s2t_model_file   = f"{model}/{model_name}"
device = "cpu"

s2t_model, s2t_train_args = S2TTask.build_model_from_file(
        s2t_train_config, s2t_model_file, device
)

# function to show percentage of total norm
def print_norm_procent(A, A_ort, A_par, is_A=True):
    n = torch.norm(A, p='fro').item()
    n_ort = torch.norm(A_ort, p='fro').item()
    n_par = torch.norm(A_par, p='fro').item()
    perc_ort = 100.0 * n_ort / n
    perc_par = 100.0 * n_par / n
    logging.info(f"Change of {'A' if is_A else 'B'}: in parallel space = {perc_par:.2f}; in orthogonal space = {perc_ort:.2f}")
    return perc_ort, perc_par

tot_perc_ort, tot_perc_par, num_layers = 0.0, 0.0, 0
# multiply E for new task with eta
for _, module in s2t_model.named_modules():
    if isinstance(module, LoRALayer):
        A2, B2 = module.get_current_adapter()[:2]
        A , B  = module.get_adapter(module.lang_sym, 0)[:2]
        # 1. analyse and average A
        A_is_zero = (A2.data == 0).sum() == A2.numel()
        if A_is_zero:
            logging.info(f"A contains only zeros: averaging has no point")
        else:
            Q, R = torch.linalg.qr(A.T)  # Q: (d, r)
            Q = Q.T # Q: (r, d)
            P_Q = Q.T @ Q  # P_Q: (d, d)
            A2_par = A2 @ P_Q
            A2_ort = A2 - A2_par
            # print stats
            perc_ort, perc_par = print_norm_procent(A2, A2_ort, A2_par, True) 
            tot_perc_ort, tot_perc_par = tot_perc_ort + perc_ort, tot_perc_par + perc_par
            num_layers += 1
            A_data = A.data + eta_a * A2_par + eta_b * A2_ort   
            A.data.copy_(A_data)
            A2.data.copy_(torch.zeros_like(A_data))
        # 2. analyse and average B
        if B2 is not None:
            Q, R = torch.linalg.qr(B)  # Q: (d, r)
            P_Q = Q @ Q.T  # P_Q: (d, d)
            B2_par = P_Q @ B2
            B2_ort = B2 - B2_par
            print_norm_procent(B2, B2_ort, B2_par, False)
            B_data = B.data + eta_a * B2_par + eta_b * B2_ort
            B.data.copy_(B_data)
            B2.data.copy_(torch.zeros_like(B_data))
        else:
            logging.info(f"B is None: averaging has no point")
        # update A, B
        A.data.copy_(A_data)
        #if B2 is not None:
        #    B.data.copy_(B_data)
        # update A2, B2
        A2.data.copy_(torch.zeros_like(A_data))
        #if B2 is not None:    
        #    B2.data.copy_(torch.zeros_like(B_data))
tot_perc_ort, tot_perc_par = tot_perc_ort / max(1, num_layers), tot_perc_par / max(1, num_layers)
logging.info(f"Total Change of A: in parallel space = {tot_perc_par:.2f}; in orthogonal space = {tot_perc_ort:.2f}")
# get state_dict
new_state_dict = s2t_model.state_dict()

logger.info("Storing the model.. (at %s/%s" % (args.model, name))

torch.save(new_state_dict, "%s/%s" % (args.model, name))


logger.info("")
logger.info("-" * 100)

