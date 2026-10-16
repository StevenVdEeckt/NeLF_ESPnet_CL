from espnet2.tasks.s2t import S2TTask
import torch
import logging
from torch import nn
from decimal import Decimal
import os

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

s2t_train_config = "exp/ml/s2t_train_asr_owsmv31_base_raw_pretrained_bpe50000/config.yaml"
s2t_model_file   = "exp/ml/s2t_train_asr_owsmv31_base_raw_pretrained_bpe50000/valid.acc.ave.pth"
device = "cpu"

s2t_model, s2t_train_args = S2TTask.build_model_from_file(
        s2t_train_config, s2t_model_file, device
)
logging.info("Done loading!")

# alpha for SVD
alpha=99.5
# layers to skip
skip_layers = ['ctc', 'output', 'encoder.embed', 'merge_proj']
# to keep track of layer number
layer_num = 0
# dict with free rank per layer
free_rank = load_free_rank_dict(s2t_train_config)
current_free_rank_dict = {}

for name, param in s2t_model.named_modules():
    if isinstance(param, nn.Linear) and not_present(name, skip_layers):
        layer_num += 1
        # Get the weight matrix
        weight = param.weight.data

        # Apply SVD
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

        # Calculate the total "power" in the singular values (sum of squared singular values)
        total_power = torch.sum(S ** 2).item()

        # Accumulate singular values until reaching the desired percentage of total power
        running_power = 0.0
        k = 0  # The number of singular values to keep
        for i in range(len(S)):
            running_power += (S[i] ** 2).item()
            k += 1
            if running_power / total_power >= alpha / 100.0:
                break

        # Reconstruct the weight matrix by keeping only the top-k singular values
        S_reconstructed = torch.zeros_like(S)
        S_reconstructed[:k] = S[:k]  # Keep the top-k singular values based on power

        # Reconstruct the weight matrix
        weight_reconstructed = U @ torch.diag_embed(S_reconstructed) @ Vh
        reconstruction_error = Decimal(torch.norm(weight - weight_reconstructed).item())

        # Update the weight matrix in the model
        param.weight.data = weight_reconstructed

        # Log the percentage of total power kept
        percent_power_kept = (running_power / total_power) * 100
        logging.info(f"({layer_num}) {name}: kept {percent_power_kept:.2f}% of the total power with {k} out of {S.size(0)} ({100.0 * k / S.size(0):.2f}%) singular values (reconstruction error = {reconstruction_error:.2E}).")

        # Update free rank dict
        current_free_rank_dict[name] = S.size(0) - k

# Save the updated model with SVD-applied weights
alpha_name = str(alpha).replace(".", "")
output_model_file = f"exp/ml/s2t_train_asr_owsmv31_base_raw_pretrained_bpe50000/valid.acc.ave.svd{alpha_name}.pth"
torch.save(s2t_model.state_dict(), output_model_file)
logging.info(f"Model saved with {alpha}% of the total power in {output_model_file}")

# save free rank dict
free_rank[alpha] = current_free_rank_dict
save_free_rank_dict(free_rank, s2t_train_config)

