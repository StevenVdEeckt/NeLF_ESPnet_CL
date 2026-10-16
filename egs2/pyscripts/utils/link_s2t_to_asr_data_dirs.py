import os
import shutil

def setup_asr_directory(s2t_dir, asr_dir, link_directories=False):
    print("Setting up ASR directories: \n --> S2T: %s \n --> ASR: %s" % (s2t_dir, asr_dir))
    if link_directories:
        os.symlink(s2t_dir, asr_dir)
    else:
        # Ensure the ASR directory exists
        os.makedirs(asr_dir, exist_ok=True)

        # List all files in the S2T directory and create symbolic links
        for item in os.listdir(s2t_dir):
            s2t_item_path = os.path.join(s2t_dir, item)
            asr_item_path = os.path.join(asr_dir, item)

            # If it's the special case of text.ctc, link it as text in ASR dir
            if item == 'text.ctc':
                os.symlink(s2t_item_path, os.path.join(asr_dir, 'text'))
            elif item != 'text' and item != 'text.prev':  # Skip S2T-specific files
                # For all other files, create a symlink in ASR directory
                os.symlink(s2t_item_path, asr_item_path)

# Example usage
datasets = ['cgn/train_cgnvl', 'cgn/train_cgnnl', 'cgn/dev_cgnvl', 'cgn/dev_cgnnl', 'cgn/test_cgnvl', 'cgn/test_cgnnl']
base_path = '/esat/spchtemp/spchdisk_orig/svandere/espnet/asr/ml/dump/raw/'  # Adjust this path to your datasets location

for dataset in datasets:
    # dump/raw
    s2t_dir = os.path.join(base_path, 'owsm', dataset)
    asr_dir = os.path.join(base_path, dataset)
    setup_asr_directory(s2t_dir, asr_dir)
    # dump/raw/org
    s2t_dir = os.path.join(base_path, "org", "owsm", dataset)
    asr_dir = os.path.join(base_path, "org", dataset)
    setup_asr_directory(s2t_dir, asr_dir, link_directories=True)
