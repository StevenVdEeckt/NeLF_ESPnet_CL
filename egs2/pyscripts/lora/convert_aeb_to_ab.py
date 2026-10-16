import torch
import argparse
import shutil
import yaml
import os

# Step 0: Read command line arguments
#   - old_dir: the directory of the old model
#   - new_dir: the directory of the new model
parser = argparse.ArgumentParser()
parser.add_argument("--old_dir", type=str, required=True, help="Directory of the old model")
parser.add_argument("--new_dir", type=str, required=True, help="Directory to store the new model")
args = parser.parse_args()

# Step 1: Prepare the old model
# name of the model
old_model = f"{args.old_dir}/latest.pth"
# load the model
old_model = torch.load(old_model)
# for the new model
new_model = {}
# B*E*A --> B*A
for name, param in old_model.items():
    if 'lora_A' in name:
        A = param
        B = old_model[name.replace('lora_A', 'lora_B')]
        E = old_model[name.replace('lora_A', 'lora_E')].squeeze()
        # B <- B*sqrt(E) ; A <- sqrt(E)*A
        assert (E < 0).sum() == 0, f"E has negative values: {E[E < 0]}"
        sqrt_E = torch.sqrt(E)
        B_new = B @ torch.diag(sqrt_E)
        A_new = torch.diag(sqrt_E) @ A
        # store in new state dict
        new_model[name] = A_new
        new_model[name.replace('lora_A', 'lora_B')] = B_new
    elif not ('lora_E' in name or 'lora_B' in name):
        new_model[name] = param
# store the new model
os.makedirs(args.new_dir, exist_ok=True)
torch.save(new_model, f"{args.new_dir}/latest.pth")

# Step 2: Copy f"{args.old_dir}/rank_pattern.pth" to f"{args.new_dir}/rank_pattern.pth"
rank_pattern_src = f"{args.old_dir}/rank_pattern.pth"
rank_pattern_dst = f"{args.new_dir}/rank_pattern.pth"
shutil.copy(rank_pattern_src, rank_pattern_dst)

# Step 3: Copy and update f"{args.old_dir}/config.yaml
# load the old yaml file
config_src = f"{args.old_dir}/config.yaml"
config_dst = f"{args.new_dir}/config.yaml"
with open(config_src, "r") as f:
    config = yaml.safe_load(f)

# remove "adapter_allocator" key from yaml
config.pop("adapter_allocator", None)

# remove "adapter_allocator_conf" key from yaml
config.pop("adapter_allocator_conf", None)

# change "adapter" key to clsvdlora2
config["adapter"] = "clsvdlora2"

# in "adapter_conf" (a dictionary), remove "minimal_rank", "alpha_svd"
if "adapter_conf" in config and isinstance(config["adapter_conf"], dict):
    config["adapter_conf"].pop("minimal_rank", None)
    config["adapter_conf"].pop("alpha_svd", None)

# change output dir from old to new
config["output_dir"] = args.new_dir

# store new yaml fiile at f"{args.new_dir}/config.yaml
with open(config_dst, "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)

# DONE!
print("✅ Done!")
