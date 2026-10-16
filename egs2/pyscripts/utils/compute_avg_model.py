import torch
import numpy as np
import argparse
import logging

parser = argparse.ArgumentParser(description='Computing the average model...')
parser.add_argument('model',  type=str, help='Path to the model directory')
parser.add_argument('name', type=str, help="Name of the model to store")
parser.add_argument('--model_name', type=str, default="valid.acc.ave.pth", help="Name of the model (default = valid.acc.ave.pth")
parser.add_argument('--old_model', type=str, default="", help="Specify old model if not the initial model from 'model'")
parser.add_argument('--avg_strategy', type=str, default='all', help="Determines how to compute average model. Can be a defined strategy or a comma-separated list of blocks that should be averaged. E.g. 'encoder', 'all', or 'encoder.encoders.3,encoder.encoders.4'.")
parser.add_argument('--eta', type=float, default=0.5, help="Eta, to compute the averaging (eta must be a float and 0 <= eta <= 1)")
parser.add_argument('--weights_old', type=str, default="", help="Weight of the old model, stored as dictionary")
parser.add_argument('--weights_new', type=str, default="", help="Weight of the new model, stored as dictionary")

args = parser.parse_args()

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger('main')

# set some params
model = args.model
model_name = args.model_name
old_model = args.old_model if args.old_model else "%s/initial_model.pth" % model
avg_strategy = args.avg_strategy
weights_old = args.weights_old
weights_new = args.weights_new
eta = args.eta
assert 0 <= eta <= 1
name = args.name

# printing information
logger.info("Computing the AVERAGE MODEL..")
logger.info("Loading the model from: %s" % model)
logger.info("... will use the model: %s" % model_name)
logger.info("... old model will be: %s" % old_model)
logger.info("... strategy will be: %s" % avg_strategy)
logger.info("... eta will be: %.3f" % eta)
if weights_old and weights_new:
    logger.info("... weights for old model: %s" % weights_old)
    logger.info("... weights for the new model: %s" % weights_new)


# load the 'new' model
final_state_dict = torch.load('%s/%s' % (model, model_name), map_location='cpu')
# load the 'initial' model
initial_state_dict = torch.load(old_model, map_location='cpu')
eps=1e-10

weights = None

# loading the weights (if any)
if weights_new and not weights_old:
   assert isinstance(weights_new, str)
   weights = torch.load(weights_new, map_location='cpu')

elif weights_old and weights_new:
   wold = torch.load(weights_old, map_location='cpu')
   wnew = torch.load(weights_new, map_location='cpu')

   weights = {k: wnew[k] / (wnew[k] + wold[k] + eps) for k in wnew.keys()}


if weights is not None:
    missing_layers = [k for k, v in final_state_dict.items() if not k in weights.keys()]
    logger.info("Following layers are missing: %s" % str(missing_layers))
    for layer in missing_layers:
        weights[layer] = 0.5

else:
    logger.info("No weights provided -- using 0.50")
    weights = {k: 0.5 * torch.ones_like(v) for k, v in final_state_dict.items()}

# avg_strategies: lambda def that returns True if average MUST BE COMPUTED
avg_strategies = {
        'all': lambda name: True,
        'encoder': lambda name: 'encoder' in name,
        'encoder+ctc': lambda name: 'encoders' in name or 'ctc' in name,
        'embed': lambda name: 'encoder' in name and 'embed' in name,
        'encoder.feedforward': lambda name: 'encoder' in name and 'feed_forward' in name,
        'encoder.attention': lambda name: 'encoder' in name and 'attn' in name,
        'encoder.conv': lambda name: 'encoder' in name and 'conv' in name,
        'encoder.feedforward+conv': lambda name: 'encoder' in name and ('feed_forward' in name or 'conv' in name),
        'encoder.feedforward+attention': lambda name: 'encoder' in name and ('feed_forward' in name or 'attn' in name),
        'encoder.conv+attention': lambda name: 'encoder' in name and ('conv' in name or 'attn' in name),
        'encoder.6_11': lambda name: 'encoders' in name and int(name.split(".")[2]) > 5,
        'encoder.6_9': lambda name: 'encoders' in name and int(name.split(".")[2]) > 5 and int(name.split(".")[2]) < 10,
        'not encoder.9_11': lambda name: not ('encoders' in name and int(name.split(".")[2]) > 8),
        'encoder.feedforward.6_11': lambda name: 'encoder' in name and 'feed_forward' in name and int(name.split(".")[2]) > 5,
        'norm': lambda name: 'norm' in name,
        'attention': lambda name: 'attn' in name,
        'decoder.attention': lambda name: 'decoder' in name and 'attn' in name,
        'output': lambda name: 'out' in name or 'ctc' in name,
        'initial_svr': lambda name: 'encoder' in name and ('bias' in name or 'norm' in name or 'conv' in name),
        'not_output': lambda name: not (('decoder' in name and ('embed' in name or 'output' in name)) or 'ctc' in name),
        'decoder': lambda name: 'decoder' in name,
        'decoder+output': lambda name: 'decoder' in name or 'ctc' in name or 'out' in name,
        'no-adapter': lambda name: not 'adapter' in name,
                 }

# select the right strategy
to_exclude = ['frontend', 'normalize', 'running_var', 'running_mean', 'num_batches_tracked']
if len(avg_strategy.split(",")) > 1:
    strategy = lambda x: any(x.startswith(s) for s in avg_strategy.split(',')) and not any(e in x for e in to_exclude)
else:
    strategy = lambda x: avg_strategies[avg_strategy](x) and not any(e in x for e in to_exclude)
# number of layers to compute average of
logger.info("Compute average of %d out of %d layers (with strategy = %s)" % 
        (len([n for n, _ in final_state_dict.items() if strategy(n)]), len(final_state_dict), strategy))

logger.info(f"Layers: {[n for n, _ in final_state_dict.items() if strategy(n)]}")

logger.info("Computing the average model..")

avg_layer = lambda o, n, w, eta: (((1 - eta) * (1 - w) * o + eta * w * n) / ((1 - eta) * (1 - w) + eta * w + eps))

new_state_dict = {n: (avg_layer(initial_state_dict[n], final_state_dict[n], weights[n], eta) if strategy(n) else final_state_dict[n]) for n in final_state_dict.keys()}

logger.info("Storing the model.. (at %s/%s" % (args.model, name))

torch.save(new_state_dict, "%s/%s" % (args.model, name))


logger.info("")
logger.info("-" * 100)

