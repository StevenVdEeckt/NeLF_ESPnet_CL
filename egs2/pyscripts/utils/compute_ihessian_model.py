import torch
import argparse
import logging
import os

# Argument Parser
parser = argparse.ArgumentParser(description='Computing the Hessian-based average model...')
parser.add_argument('model', type=str, help='Path to the model directory')
parser.add_argument('name', type=str, help="Name of the model to store")
parser.add_argument('--model_name', type=str, default="valid.acc.ave.pth", help="Name of the model (default = valid.acc.ave.pth)")
parser.add_argument('--old_model', type=str, default="", help="Specify old model if not the initial model from 'model'")
parser.add_argument('--avg_strategy', type=str, default='all', help="Strategy for averaging layers.")
parser.add_argument('--eta', type=float, default=1.0, help="Eta, the 'regularization weight'")
parser.add_argument('--alpha', type=float, default=1.0, help="Alpha, for computing the averaging between H and I (0 <= alpha <= 1)")
parser.add_argument('--beta', type=float, default=1.0, help="Beta, for computing the averaging between old and new (0 <= beta <= 1)")
parser.add_argument('--hessian', type=str, required=True, help="Hessian file containing S, Y, B0 tensors")

args = parser.parse_args()

# Logging
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger('main')

# Set parameters
model_path = args.model
old_model_path = args.old_model if args.old_model else os.path.join(model_path, "initial_model.pth")
hessian_path = args.hessian
eta = args.eta
alpha = args.alpha
beta = args.beta
logging.info(f"Using eta = {eta}, alpha = {alpha}, beta = {beta}")
#assert 0 <= eta <= 1, "Eta must be between 0 and 1."

logger.info("Loading models...")
final_state_dict = torch.load(os.path.join(model_path, args.model_name), map_location='cpu')
initial_state_dict = torch.load(old_model_path, map_location='cpu')


logger.info("Loading Hessian from: %s" % hessian_path)
hessian_data = torch.load(hessian_path, map_location='cpu')[-1]  # Select last Hessian entry
S, Y, B0, exclude_layers = hessian_data['S'], hessian_data['Y'], hessian_data['ewc'], hessian_data['exclude']

# add to exclude layers: frontend, normalize, running_mean, running_var, num_batches_tracked
remove_layers = ['frontend', 'normalize', 'running_mean', 'running_var', 'num_batches_tracked']
for name, _ in final_state_dict.items():
    if any(r in name for r in remove_layers):
        exclude_layers.append(name)

logger.info("Computing the Hessian-based average model...")
eps = 1e-10  # Avoid division errors

# Define strategies
avg_strategies = {
    'all': lambda name: True,
    'encoder': lambda name: 'encoder' in name,
    'encoder+ctc': lambda name: 'encoders' in name or 'ctc' in name,
    'embed': lambda name: 'encoder' in name and 'embed' in name,
    'encoder.feedforward': lambda name: 'encoder' in name and 'feed_forward' in name,
    'encoder.attention': lambda name: 'encoder' in name and 'attn' in name,
    'encoder.conv': lambda name: 'encoder' in name and 'conv' in name,
    'decoder': lambda name: 'decoder' in name,
    'output': lambda name: 'out' in name or 'ctc' in name,
}

# Parse strategy
if "," in args.avg_strategy:
    strategy = lambda x: any(x.startswith(s) for s in args.avg_strategy.split(",")) and x not in exclude_layers
else:
    strategy = lambda x: avg_strategies.get(args.avg_strategy, lambda _: False)(x) and x not in exclude_layers

# Select layers to average
layers_to_avg = {n for n in final_state_dict.keys() if strategy(n)}

logger.info("Averaging %d layers (out of %d) with strategy = %s" % (
    len(layers_to_avg), len(final_state_dict), args.avg_strategy))

for name, _ in final_state_dict.items():
    if name in layers_to_avg:
        print(name)

# Compute update
delta_theta = {n: final_state_dict[n] - initial_state_dict[n] for n in layers_to_avg}

# Convert parameter dictionaries to vectors
def to_vec(param_dict):
    return torch.cat([p.reshape(-1) for p in param_dict.values()])

def to_param(x, sizes):
    """
    Convert a 1D tensor back into a state_dict.
    """
    sd, k = {}, 0
    for n, p in sizes.items():
        sd[n] = x[k:k + p['numel']].view(p['size'])
        k += p['numel']
    return sd

# Convert Hessian vectors to param dicts and remove non-matching layers
def filter_hessian(S, Y, B0, layers_to_keep):
    """
    Filters Hessian tensors to only contain layers that match `layers_to_keep`
    """
    params_sizes = {n: {"numel": p.numel(), "size": p.size()} for n, p in final_state_dict.items() if not n in exclude_layers}

    # Convert to full param dicts
    S_dict = [to_param(S[:, i], params_sizes) for i in range(S.size(1))]
    Y_dict = [to_param(Y[:, i], params_sizes) for i in range(Y.size(1))]
    B0_dict = to_param(B0, params_sizes)

    # Keep only relevant layers
    S_filtered = [{k: _S_dict[k] for k in layers_to_keep} for _S_dict in S_dict]
    Y_filtered = [{k: _Y_dict[k] for k in layers_to_keep} for _Y_dict in Y_dict]
    B0_filtered = {k: B0_dict[k] for k in layers_to_keep}

    # Turn to vec again
    S = torch.cat([to_vec(Si).view(-1, 1) for Si in S_filtered], dim=1)
    Y = torch.cat([to_vec(Yi).view(-1, 1) for Yi in Y_filtered], dim=1)

    return S, Y, to_vec(B0_filtered)

# Filter Hessian tensors
S_filtered, Y_filtered, B0_filtered = filter_hessian(S, Y, B0, layers_to_avg)

# Compute Hessian-inverted update
def InverseHessianVectorProduct_BFGS(v, S, Y, B0, eps_ewc=1e-4, lambda_reg=1e-3):
    """
    Computes H^-1 v using BFGS inverse approximation.

    Args:
        v (torch.Tensor): Input vector.
        S (torch.Tensor): Matrix of parameter changes.
        Y (torch.Tensor): Matrix of gradient changes.
        B0 (torch.Tensor): Initial Hessian approximation.

    Returns:
        torch.Tensor: Approximated inverse Hessian-vector product.
    """
    # remove all pairs (s,y) for which s'y < 0
    if S is not None and Y is not None:
        selected_indices = []
        for i in range(S.shape[1]):
            sy = torch.dot(S[:, i], Y[:, i])
            if sy >= 0:
                selected_indices.append(i)
        S, Y = S[:, selected_indices], Y[:, selected_indices]

    logging.info(f"Size of S = {S.size()}, Y = {Y.size()}")

    if S is None or Y is None:
        return v  # No memory, return unmodified vector

    q = v.clone()
    alphas = []

    # Forward pass
    for i in range(S.shape[1]):
        rho = 1 / (torch.dot(S[:, i], Y[:, i]) + eps)
        alpha = rho * torch.dot(S[:, i], q)
        alphas.append(alpha)
        q -= alpha * Y[:, i]

    # Scaling with B0 estimate
    #if B0 is None:
    #print(f"Gammas = {[torch.dot(S[:, i], Y[:, i]) / torch.dot(Y[:, i], Y[:, i]).item() for i in range(S.size(1))]}")
    gamma = sum([torch.dot(S[:, i], Y[:, i]) / torch.dot(Y[:, i], Y[:, i]).item() for i in range(S.size(1))]) / S.size(1)
    #print(f"Gamma = {gamma}")
    #else:
    #gamma = lambda_reg / (B0 + eps_ewc * B0.max())
    #gamma = 1e-3 / (B0 + eps)
    logging.info(f"B0... min={gamma.min()}, max={gamma.max()}, mean={gamma.mean()}")
    r = gamma * q
    #r = 0 * q
    #r = q
    # Backward pass
    for i in range(S.shape[1] - 1, -1, -1):
        rho = 1 / (torch.dot(S[:, i], Y[:, i]) + eps)
        beta = rho * torch.dot(Y[:, i], r)
        r += S[:, i] * (alphas[i] - beta)

    return r

# Compute Hessian-preconditioned update
delta_theta_vec = to_vec(delta_theta)
hessian_update = (1 - alpha) * eta * InverseHessianVectorProduct_BFGS(delta_theta_vec, S_filtered, Y_filtered, B0_filtered) + alpha * delta_theta_vec
hessian_update = beta * hessian_update
print(f"Norm of update.. before = {torch.norm(delta_theta_vec):.2f}, after = {torch.norm(hessian_update):.2f}")
hessian_update = hessian_update / torch.norm(hessian_update) * torch.norm(delta_theta_vec)
theta2_updated_vec = to_vec({k: v for k, v in initial_state_dict.items() if k in layers_to_avg}) + hessian_update

# Convert back to parameter dictionary
theta2_updated_dict = to_param(theta2_updated_vec, {n: {"numel": p.numel(), "size": p.size()} for n, p in initial_state_dict.items() if n in layers_to_avg})

# Replace only averaged layers
for name in layers_to_avg:
    final_state_dict[name] = theta2_updated_dict[name]

# Save the updated model
torch.save(final_state_dict, os.path.join(model_path, args.name))
logger.info("Model saved at: %s/%s" % (model_path, args.name))

