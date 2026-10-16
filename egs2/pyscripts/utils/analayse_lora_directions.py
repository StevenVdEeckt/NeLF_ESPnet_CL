from espnet2.tasks.s2t import S2TTask
import torch
import logging
from torch import nn
from decimal import Decimal
import os
from espnet2.layers.clloralib import LoRALayer

def not_present(name, list_of_strings):
    for s in list_of_strings:
        if s in name:
            return False
    return True

def load_free_rank_dict(config_file):
    file_name = config_file.replace("config.yaml", "svd_free_rank.pth")
    if os.path.isfile(file_name):
        return torch.load(file_name)
    return {}

def save_free_rank_dict(free_rank_dict, config_file):
    file_name = config_file.replace("config.yaml", "svd_free_rank.pth")
    torch.save(free_rank_dict, file_name)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

s2t_train_config = "exp/ml/s2t_train_asr_owsmv31_base_cllora_highlr_r8_uniform_raw_cgnnl_bpe50000/config.yaml"
s2t_model_file   = "exp/ml/s2t_train_asr_owsmv31_base_cllora_highlr_r8_uniform_raw_cgnnl_bpe50000/valid.acc.ave.pth"
device = "cpu"

s2t_model, s2t_train_args = S2TTask.build_model_from_file(
        s2t_train_config, s2t_model_file, device
)
logging.info("Done loading!")

for name, param in s2t_model.named_modules():
    if isinstance(param, LoRALayer):
        # Get the weight matrix
        weight = param.weight.data

        # Apply SVD
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        
        n = max(weight.size(0), weight.size(1))
        map_score = lambda x: 100.0 * (x - torch.min(S))/(torch.max(S) - torch.min(S))

        # Get the current lora
        A, B, _ = param.get_adapter(param.lang_sym, param.task_label)

        # Compute W'' = B @ A (LoRA update)
        W_prime_prime = B @ A

        # Apply SVD to W'' = B @ A to get U'', S'', Vh'' (Vh'' is V'' transpose)
        U_prime_prime, S_prime_prime, Vh_prime_prime = torch.linalg.svd(W_prime_prime, full_matrices=False)

        # Compare V'' with V by projecting V'' onto V to get the coordinates of V'' in the basis of V
        # Note: Vh and Vh_prime_prime represent V and V'', respectively, in transposed form
        Vh_comparison = Vh.T @ Vh_prime_prime

        # Compare U'' with U: Project U'' onto U to get the coordinates of U'' in terms of U
        U_comparison = U @ U_prime_prime.T

        # Scale the comparisons by the singular values
        if weight.size(0) > weight.size(1):
            total_score = 0
            # Pad S to match the size of U
            if Vh.size(1) > S.size(0):
                S_padded = torch.cat([S, torch.zeros(Vh.size(0) - S.size(0), device=S.device)])
            else:
                S_padded = S
            for i in range(A.size(0)):
                vi = torch.diag(S_padded) @ Vh_comparison[:,i]
                score = torch.sqrt(torch.sum(torch.square(vi)))
                #logging.info(f"Component {i} of V has score {map_score(score):.2f}% ...")
                total_score += S_prime_prime[i] * score
            total_score = total_score / torch.sum(S_prime_prime)
            logging.info(f"{name}:  Total weighted-averaged score for V = {total_score:.2f}%")

            total_score = 0
            # Pad S to match the size of U
            if U.size(0) > S.size(0):
                S_padded = torch.cat([S, torch.zeros(U.size(0) - S.size(0), device=S.device)])
            else:
                S_padded = S
            for i in range(B.size(1)):
                ui = torch.diag(S_padded) @ U_comparison[i, :]
                score = torch.sqrt(torch.sum(torch.square(ui)))
                #logging.info(f"Component {i} of U has score {map_score(score):.2f}% ...")
                total_score += S_prime_prime[i] * score
            total_score = total_score / torch.sum(S_prime_prime)
            logging.info(f"{name}: Total weighted-averaged score for U = {total_score:.2f}%")

