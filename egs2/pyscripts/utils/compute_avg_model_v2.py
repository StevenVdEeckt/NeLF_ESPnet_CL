import torch
import numpy as np
import argparse
import logging

parser = argparse.ArgumentParser(description='Computing the average model...')
parser.add_argument('model',  type=str, help='Path to the model directory')
parser.add_argument('name', type=str, help="Name of the model to store")
parser.add_argument('--model_name', type=str, default="valid.acc.ave.pth", help="Name of the model (default = valid.acc.ave.pth")
parser.add_argument('--old_model', type=str, default="", help="Specify old model if not the initial model from 'model'")
parser.add_argument('--eta', type=float, default=0.5, help="Eta, to compute the averaging (eta must be a float and 0 <= eta <= 1)")

args = parser.parse_args()

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger('main')

# set some params
model = args.model
model_name = args.model_name
old_model = args.old_model if args.old_model else "%s/initial_model.pth" % model
eta = args.eta
assert 0 <= eta
name = args.name

# printing information
logger.info("Computing the AVERAGE MODEL..")
logger.info("Loading the model from: %s" % model)
logger.info("... will use the model: %s" % model_name)
logger.info("... old model will be: %s" % old_model)
logger.info("... eta will be: %.3f" % eta)
# load the 'new' model
final_state_dict = torch.load('%s/%s' % (model, model_name), map_location='cpu')
# load the 'initial' model
initial_state_dict = torch.load(old_model, map_location='cpu')


# select the right strategy
to_exclude = ['frontend', 'normalize', 'running_var', 'running_mean', 'num_batches_tracked']
strategy = lambda x: not any(e in x for e in to_exclude)
# number of layers to compute average of
logger.info("Compute average of %d out of %d layers (with strategy = %s)" % 
        (len([n for n, _ in final_state_dict.items() if strategy(n)]), len(final_state_dict), strategy))

logger.info(f"Layers: {[n for n, _ in final_state_dict.items() if strategy(n)]}")

logger.info("Computing the average model..")

new_state_dict = {}

for n, param in final_state_dict.items():
    if not strategy(n):
        new_state_dict[n] = param
    else:
        init_param = initial_state_dict[n]
        update = param - init_param
        new_state_dict[n] = init_param + eta * update

logger.info("Storing the model.. (at %s/%s" % (args.model, name))

torch.save(new_state_dict, "%s/%s" % (args.model, name))


logger.info("")
logger.info("-" * 100)

