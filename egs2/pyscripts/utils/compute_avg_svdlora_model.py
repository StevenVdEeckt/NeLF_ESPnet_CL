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
parser.add_argument('--eta_a', type=float, default=0.5, help="Eta_a, to compute the averaging (eta_a must be a float and 0 <= eta_a <= 1)")
parser.add_argument('--eta_b', type=float, default=0.5, help="Eta_b, to compute the averaging (eta_b must be a float and 0 <= eta_b <= 1)")


args = parser.parse_args()

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger('main')

# set some params
model = args.model
model_name = args.model_name
eta_a, eta_b = args.eta_a, args.eta_b
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

# multiply E for new task with eta
for _, module in s2t_model.named_modules():
    if isinstance(module, LoRALayer):
        A2, B2 = module.get_current_adapter()[:2]
        A , B  = module.get_adapter(module.lang_sym, 0)[:2]
        # average A
        A_is_zero = (A2.data == 0).sum() == A2.numel()
        if A_is_zero:
            logging.info(f"A contains only zeros: averaging has no point")
        A_data = A.data + eta_a * A2.data
        # average B
        if B2 is not None:
            B_data = B.data + eta_b * B2.data
        else:
            logging.info(f"B is None: averaging has no point")
        # update A, B
        A.data.copy_(A_data)
        if B2 is not None:
            B.data.copy_(B_data)
        # update A2, B2
        A2.data.copy_(torch.zeros_like(A_data))
        if B2 is not None:    
            B2.data.copy_(torch.zeros_like(B_data))

# get state_dict
new_state_dict = s2t_model.state_dict()

logger.info("Storing the model.. (at %s/%s" % (args.model, name))

torch.save(new_state_dict, "%s/%s" % (args.model, name))


logger.info("")
logger.info("-" * 100)

