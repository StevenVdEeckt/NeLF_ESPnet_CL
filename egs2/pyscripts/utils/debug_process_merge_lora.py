from espnet2.tasks.s2t import S2TTask
import torch
import logging
from torch import nn
from decimal import Decimal
import os
from espnet2.layers.clloralib import LoRALayer

def update_projection_matrix(projection_matrices, name, P_A_new):
    if projection_matrices is None:
        projection_matrices = {}
    P_A = projection_matrices.get(name)
    if P_A is None:
        return P_A_new.detach()
    else:
        return P_A + P_A_new.detach().to(P_A.device)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

s2t_model_dir = "exp/ml/s2t_train_asr_owsmv31_base_clssvdlora_svd090_s1em2_minr4_rao_reg2f_1e4_svder20_raw_cgnvl_from_cgnnl_bpe50000"
s2t_model_file = "unpruned_model.pth"
s2t_control_model_file = "tot_pruned_model.pth"

projection_matrices = "exp/ml/s2t_train_asr_owsmv31_base_clssvdlora_svd090_s1em2_minr4_raw_cgnnl_bpe50000/projection_matrices.pth"

lang, task = 93, 1

device = "cpu"

s2t_model, s2t_train_args = S2TTask.build_model_from_file(
        f"{s2t_model_dir}/config.yaml", f'{s2t_model_dir}/{s2t_model_file}', device
)
logging.info("Done loading!")

# process model
tot_rank, tot_params = 0, 0
projection_matrices = torch.load(projection_matrices, map_location='cpu')
for name, module in s2t_model.named_modules():
    if isinstance(module, LoRALayer):
        k, P_A = module.post_process_task()
        tot_params += module.in_features * k + k + module.out_features * k
        tot_rank += k
        module.get_rank_pattern()
        update_projection_matrix(projection_matrices, name, P_A)

# check if equal to control model
control_model = torch.load(f'{s2t_model_dir}/{s2t_control_model_file}', map_location=device)
model = s2t_model.state_dict()




for name, module in s2t_model.named_modules():
    if isinstance(module, LoRALayer):
        adapter_key = f'adapter{lang}_{task}'
        lora_A, lora_E, lora_B = control_model[f'{name}.{adapter_key}_lora_A'], control_model[f'{name}.{adapter_key}_lora_E'], control_model[f'{name}.{adapter_key}_lora_B']
        A, E, B = model[f'{name}.{adapter_key}_lora_A'], model[f'{name}.{adapter_key}_lora_E'], model[f'{name}.{adapter_key}_lora_B']
        # Check 1: rank is the same
        rank_diff = abs(E.squeeze().size(0) - lora_E.squeeze().size(0))
        # Check 2: difference in E
        #E_diff = torch.norm(module.get_adapter(lang, task)[1] - lora_E, p='fro').item()
        E_diff = torch.norm(E - lora_E, p='fro').item()
        # Check 3: difference in A and B
        #A_diff = torch.norm(module.get_adapter(lang, task)[0] - lora_A, p='fro').item()
        #B_diff = torch.norm(module.get_adapter(lang, task)[2] - lora_B, p='fro').item()
        A_diff = torch.norm(A - lora_A, p='fro').item()
        B_diff = torch.norm(B - lora_B, p='fro').item()
        logging.info(f"{name}.. C1: rank={rank_diff} C2: E={E_diff} C3: A={A_diff}, B={B_diff}")
