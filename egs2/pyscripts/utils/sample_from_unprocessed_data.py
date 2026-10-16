import argparse
import numpy as np
import logging
import os
import shutil
import torch
import sentencepiece as spm
import torch.nn.functional as func
import inspect
import re

#######################################################################################################################
################################### HELP FUNCTIONS ####################################################################
#######################################################################################################################

def extract_wav(input_id):
    # Regular expression to match everything up to and including 'mp3'
    match = re.search(r'.*?mp3', input_id)
    if match:
        return match.group(0)  # Returns the matched part including 'mp3'
    return input_id  # Return the entire ID if 'mp3' is not found




#######################################################################################################################
################################### SAMPLING METHOD FUNCTIONS #########################################################
#######################################################################################################################

def uniform(utts, num_samples, probs=None):
    """
        Uniform sampling
    """
    return np.random.choice(utts, size=num_samples, replace=False, p=probs)


#######################################################################################################################
################################### SAMPLING_MEMORY.PY: MAIN FILE #####################################################
#######################################################################################################################

def main():

    parser = argparse.ArgumentParser(description='Sampling Memory from Training Data...')
    parser.add_argument('train_data', type=str, help='Path to training data dir')
    parser.add_argument('name', type=str, help='Name of the memory')
    parser.add_argument('--N', type=int, default=1000, help='Number of samples in the memory')
    parser.add_argument('--min_inputs', type=int, default=0, help='Minimum number of input frames in for samples in the memory')
    parser.add_argument('--min_outputs', type=int, default=0, help='Minimum number of output symbols (i.e. output length) for samples in memory')
    parser.add_argument('--sampling_method', type=str, default='uniform', choices=['uniform'], help="Sampling method, choose from ['uniform'']")
    parser.add_argument('--previous_memory', type=str, default='', help='To add a previous memory to the newly sampled memory')
    args = parser.parse_args()

    # some params
    data_dir = args.train_data
    min_outputs = args.min_outputs
    min_inputs = args.min_inputs
    strategy = args.sampling_method
    num_samples = args.N
    old_data_dir = args.previous_memory

    # Prepare log file
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')
    logger = logging.getLogger('main')

    # Files that must be read from training set
    files = ['spk2utt', 'text', 'text.ctc', 'text.prev', 'utt2spk', 'wav.scp']

    # check if valid data dir
    for f in files:
        assert os.path.isfile("%s/%s" % (data_dir, f)), "file does not exist: %s/%s" % (data_dir, f)

    # some important things
    utt2spk, utt2txt, utt2txtprev, utt2txtctc = {}, {}, {}, {}
    utt2num_frames, utt2num_out = {}, {}
    utts = []

    # Reading all files
    logger.info("Reading files...")

    logger.info("...1/ '%s/text" % data_dir)
    with open('%s/text' % data_dir, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            line = line.strip("\n").split(" ")
            utt, text = line[0], line[1:]
            utts.append(utt)
            if len(text) > min_outputs:
                utt2num_out[utt] = len(text)
                utt2txt[utt] = text
            line = f.readline()

    logger.info("...2/ '%s/utt2spk" % data_dir)

    with open('%s/utt2spk' % data_dir, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            utt, spk = line.strip("\n").split(" ")
            utt2spk[utt] = spk
            line = f.readline()

    logger.info("Done! loading the files.. Found %d utterances!" % len(utts))

    # Sampling methods
    sampling_methods = {
        'uniform': uniform,
    }

    # optional arguments for Sampling methods
    sm_args = {
        'utts': utts,
        'utt2txt': utt2txt,
        'utt2num_out': utt2num_out,
        'utt2spk': utt2spk,
        'num_samples': num_samples,
    }

    # Start sampling
    logger.info("Sampling with sampling method.. = %s" % strategy)
    # get the arguments of the sampling method
    _args = {arg: val for arg, val in sm_args.items() if arg in inspect.getfullargspec(sampling_methods[strategy]).args}
    # run the sampling method
    selected_utts = sampling_methods[strategy](**_args)

    logger.info("Done! New dataset contains %d utterances" % len(selected_utts))

    # will be needed later
    selected_spk = [utt2spk[utt] for utt in selected_utts]
    selected_num_out = [utt2num_out[utt] for utt in selected_utts]
    selected_txt = [utt2txt[utt] for utt in selected_utts]

    # saving the memory
    files_utt1 = ['text', 'utt2spk','text.ctc', 'text.prev', 'segments']
    files_utt2 = ['wav.scp']
    files_copy = []
    files_spk1 = ['spk2utt']

    new_dir = "/".join(data_dir.split("/")[:-1]) + "/%s" % args.name

    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

    logger.info("Storing memory at %s.." % new_dir)

    # first the utterance-related files
    for fname in files_utt1:
        lines = {}
        logger.info(f"Writing file... {fname}")
        with open("%s/%s" % (data_dir, fname), 'r', encoding='utf-8') as fold:
            line = fold.readline()
            while line:
                utt_ = line.split(" ")[0]
                lines[utt_] = line
                line = fold.readline()
        with open('%s/%s' % (new_dir, fname), 'w', encoding='utf-8') as fnew:
            for utt in selected_utts:
                fnew.write(lines[utt])
        
    # then the wav-related files
    for fname in files_utt2:
        lines = {}
        logger.info(f"Writing file... {fname}")
        with open("%s/%s" % (data_dir, fname), 'r', encoding='utf-8') as fold:
            line = fold.readline()
            while line:
                utt_ = line.split(" ")[0]
                lines[utt_] = line
                line = fold.readline()
        with open('%s/%s' % (new_dir, fname), 'w', encoding='utf-8') as fnew:
            for utt in selected_utts:
                fnew.write(lines[extract_wav(utt)])

    # next the copy files
    for fname in files_copy:
        logger.info(f"Copying file... {fname}")
        src, dst = '%s/%s' % (data_dir, fname), "%s/%s" % (new_dir, fname)
        shutil.copy(src, dst)

    # then the speaker-related files
    for fname in files_spk1:
        logger.info(f"Writing file... {fname}")
        lines = {}
        with open("%s/%s" % (data_dir, fname), 'r', encoding='utf-8') as fold:
            line = fold.readline()
            while line:
                spk_ = line.split(" ")[0]
                lines[spk_] = line
                line = fold.readline()
        with open('%s/%s' % (new_dir, fname), 'w', encoding='utf-8') as fnew:
            for spk in selected_spk:
                fnew.write(lines[spk])

    # optional: add samples from a previous memory
    if old_data_dir:
        for fname in files_spk1:
            fnew = open('%s/%s' % (new_dir, fname), 'a', encoding='utf-8')
            with open("%s/%s" % (old_data_dir, fname), 'r', encoding='utf-8') as fold:
                line = fold.readline()
                while line:
                    spk_ = line.split(" ")[0]
                    fnew.write(line)
                    line = fold.readline()
            fnew.close()

    logger.info("Done!")



# RUN MAIN FILE
if __name__ == "__main__":
    main()
