from espnet2.tasks.s2t import S2TTask
import torch
from espnet2.layers.clgateloralib import LoRALayer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

s2t_train_config = "exp/ml/s2t_train_asr_owsmv31_base_clgatelora_r8_uniform_debug_raw_sabed_bio_from_cgnnl_bpe50000/config.yaml"
s2t_model_file = "exp/ml/s2t_train_asr_owsmv31_base_clgatelora_r8_uniform_debug_raw_sabed_bio_from_cgnnl_bpe50000/2epoch.pth"
device = "cpu"

s2t_model, s2t_train_args = S2TTask.build_model_from_file(
        s2t_train_config, s2t_model_file, device
)

#for name, module in s2t_model.named_modules():
#    if isinstance(module, LoRALayer):
#        print(f"{name} : {module}")


logging.info("loading state_dict")
s2t_model.state_dict()

logging.info(" \n" * 10)

destination=None
prefix=""
keep_vars=False


def check_for_lora(model):
    for name, module in model._modules.items():
        if module is not None:
            if isinstance(module, LoRALayer):
                logging.info(f"Yes: {name}")
                module.state_dict(destination=destination, prefix=prefix + name + '.', keep_vars=keep_vars) 
            check_for_lora(module)
    return

state_dict = s2t_model.state_dict()

logging.info(" \n" * 10)

for name, param in state_dict.items():
    if torch.is_tensor(param):
        logging.info(f"{name} = {param.size()}")
    else:
        logging.info(f"{name} = {param}")

logging.info(" \n" * 10)

logging.info(s2t_model)

logging.info(" \n" * 10)

#s2t_model.load_state_dict(state_dict)

s2t_model.load_state_dict(state_dict)

for name, module in s2t_model.named_modules():
    if isinstance(module, LoRALayer):
        logging.info(f"{name} -> {module.task_mapping}")

