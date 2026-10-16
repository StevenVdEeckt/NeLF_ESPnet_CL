import argparse
import os
import torch

def main():
    parser = argparse.ArgumentParser(description="Sum state_dicts from specified files and save the result.")
    parser.add_argument("directory", type=str, help="Directory where the state_dict files are stored")
    parser.add_argument("filenames", type=str, help="Comma-separated list of filenames to load and sum")
    parser.add_argument("new_filename", type=str, help="Filename to save the summed state_dict to")

    args = parser.parse_args()

    # Split the comma-separated filenames and trim whitespace
    filenames = [filename.strip() for filename in args.filenames.split(',')]

    # Load the state_dicts
    state_dicts = [torch.load(os.path.join(args.directory, filename)) for filename in filenames]

    if len(state_dicts) > 1:
        # Sum the state_dicts
        summed_state_dict = sum_dicts(state_dicts)

        # Save the summed state_dict
        torch.save(summed_state_dict, os.path.join(args.directory, args.new_filename))
        print(f"Summed state_dict saved as {os.path.join(args.directory, args.new_filename)}")
    else:
        print("Only one state_dict provided. No action taken.")

def sum_dicts(dicts):
    """Sum multiple state_dicts."""
    summed_dict = dicts[0].copy()
    for key in summed_dict:
        for other_dict in dicts[1:]:
            summed_dict[key] += other_dict[key]
    return summed_dict

if __name__ == "__main__":
    main()
