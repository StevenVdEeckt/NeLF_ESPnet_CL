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
parser.add_argument('--eta', type=float, default=0.5, help="Eta, to compute the averaging (eta must be a float and 0 <= eta <= 1)")
parser.add_argument('--automatic_eta', type=bool, default=False, help="If True, eta=rank_t / sum_{i=1}^{t-1} rank_i")

args = parser.parse_args()

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger('main')

# set some params
model = args.model
model_name = args.model_name
automatic_eta = args.automatic_eta
eta = args.eta
assert 0 <= eta <= 1
name = args.name

# printing information
logger.info("Computing the AVERAGE MODEL..")
logger.info("Loading the model from: %s" % model)
logger.info("... will use the model: %s" % model_name)
logger.info("... will use automatic eta? %s" % str(automatic_eta))
logger.info("... eta will be: %.3f" % eta)

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
        lang = module.lang_sym
        new_task = module.task_label
        if automatic_eta: 
            old_rank = 0
            for t in range(new_task):
                adapter = module.get_adapter(lang, t)
                if adapter is not None:
                    old_rank += adapter[0].size(0)
            new_rank = module.get_adapter(lang, new_task)[0].size(0)
            _eta = eta * new_rank / (old_rank + new_rank)
        else:
            _eta = eta
        A, E, B = module.get_adapter(lang, new_task)[:3]
        E.data.copy_(_eta * E)

# get state_dict
new_state_dict = s2t_model.state_dict()

logger.info("Storing the model.. (at %s/%s" % (args.model, name))

torch.save(new_state_dict, "%s/%s" % (args.model, name))


logger.info("")
logger.info("-" * 100)

