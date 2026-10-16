from espnet2.tasks.s2t import S2TTask
import torch
from espnet2.layers.clloralib import LoRALayer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

s2t_train_config = "exp/ml/s2t_train_asr_owsmv31_base_clincrelora3_r8_uniform_raw_cgnnl_bpe50000/config.yaml"
s2t_model_file   = "exp/ml/s2t_train_asr_owsmv31_base_clincrelora3_r8_uniform_raw_cgnnl_bpe50000/latest.pth"
device = "cpu"

s2t_model, s2t_train_args = S2TTask.build_model_from_file(
        s2t_train_config, s2t_model_file, device
)

logging.info("Done!")

for name, param in s2t_model.named_modules():
    if isinstance(param, LoRALayer):
        print(f"layer = {name}")
        param.prune_adapters()
        #for lang_sym, (lora_A, lora_E, lora_B, ranknum, _, _) in param.adapters.items():
            #print(f"{name}, lang_sym={lang_sym} : lora_A={len(lora_A)}, lora_B={len(lora_B)}, \n lora_E={len(lora_E)}, \n ranknum={ranknum.item()}")

new_model_file = s2t_model_file.replace('latest.pth', 'pruned_latest.pth')

logging.info("Loading state dict")
state_dict = s2t_model.state_dict()

logging.info("Storing state dict")
torch.save(state_dict, new_model_file)

logging.info("Loading model from new state dict")
s2t_model, s2t_train_args = S2TTask.build_model_from_file(
        s2t_train_config, new_model_file, device
)
