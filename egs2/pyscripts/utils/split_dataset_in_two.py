import argparse
import numpy as np
import logging
import os
import shutil
import torch
import sentencepiece as spm
import torch.nn.functional as func
import inspect

#######################################################################################################################
################################### HELP FUNCTIONS ####################################################################
#######################################################################################################################


#######################################################################################################################
################################### SAMPLING_MEMORY.PY: MAIN FILE #####################################################
#######################################################################################################################

def main():

    parser = argparse.ArgumentParser(description='Sampling Memory from Training Data...')
    parser.add_argument('train_data', type=str, help='Path to training data dir')
    parser.add_argument('--split', type=float, default=0.50, help='Split of dataset - float between 0 and 1')
    args = parser.parse_args()

    # some params
    data_dir = args.train_data
    split = args.split

    # Prepare log file
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')
    logger = logging.getLogger('main')

    # Files that must be read from training set
    files = ['feats_type', 'spk2utt', 'text', 'text.ctc', 'text.prev', 'utt2num_samples', 'utt2spk', 'wav.scp']

    # check if valid data dir
    for f in files:
        assert os.path.isfile("%s/%s" % (data_dir, f)), "file does not exist: %s/%s" % (data_dir, f)

    # some important things
    utt2spk, utt2txt, utt2txtprev, utt2txtctc = {}, {}, {}, {}
    utt2num_frames, utt2num_out = {}, {}
    utts = []

    # Reading all files
    logger.info("Reading files...")

    logger.info("...1/ '%s/utt2num_samples" % data_dir)

    with open('%s/utt2num_samples' % data_dir, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            utt, num_frames = line.strip("\n").split(" ")
            num_frames = int(num_frames)
            utt2num_frames[utt] = num_frames
            line = f.readline()

    logger.info("...2/ '%s/text" % data_dir)

    with open('%s/text' % data_dir, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            line = line.strip("\n").split(" ")
            utt, text = line[0], line[1:]
            utt2num_out[utt] = len(text)
            utt2txt[utt] = text
            line = f.readline()

    logger.info("...3/ '%s/utt2spk" % data_dir)

    with open('%s/utt2spk' % data_dir, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            utt, spk = line.strip("\n").split(" ")
            utts.append(utt)
            utt2spk[utt] = spk
            line = f.readline()

    logger.info("Done! loading the files..")
    logger.info("Utterances: %d" % len(utts))


    # Start sampling
    split = int(len(utts) * args.split)
    utts1 = utts[:split]
    utts2 = utts[split:]
    assert len(utts1) + len(utts2) == len(utts)
    logger.info("Set 1 will contain %d utterances; Set 2 %d utterances" % (len(utts1), len(utts2)))

    # saving the memory
    files_utt1 = ['text', 'utt2num_samples', 'utt2spk', 'wav.scp', 'text.ctc', 'text.prev']
    files_copy = ['feats_type']
    files_spk1 = ['spk2utt']

    for i, selected_utts in enumerate([utts1, utts2], 1):
        # will be needed later
        selected_spk = [utt2spk[utt] for utt in selected_utts]
        selected_num_frames = [utt2num_frames[utt] for utt in selected_utts]
        selected_num_out = [utt2num_out[utt] for utt in selected_utts]
        selected_txt = [utt2txt[utt] for utt in selected_utts]
        
        new_dir = f"{data_dir}{i}"

        if not os.path.isdir(new_dir):
            os.makedirs(new_dir)

        logger.info("Storing dataset at %s.." % new_dir)

        # first the utterance-related files
        for fname in files_utt1:
            fnew = open('%s/%s' % (new_dir, fname), 'w', encoding='utf-8')
            with open("%s/%s" % (data_dir, fname), 'r', encoding='utf-8') as fold:
                line = fold.readline()
                while line:
                    utt_ = line.split(" ")[0]
                    if utt_ in selected_utts:
                        fnew.write(line)
                    line = fold.readline()
            fnew.close()

        # next the copy files
        for fname in files_copy:
            src, dst = '%s/%s' % (data_dir, fname), "%s/%s" % (new_dir, fname)
            shutil.copy(src, dst)

        # then the speaker-related files
        for fname in files_spk1:
            fnew = open('%s/%s' % (new_dir, fname), 'w', encoding='utf-8')
            with open("%s/%s" % (data_dir, fname), 'r', encoding='utf-8') as fold:
                line = fold.readline()
                while line:
                    spk_ = line.split(" ")[0]
                    if spk_ in selected_spk:
                        fnew.write(line)
                    line = fold.readline()
            fnew.close()
            
        logger.info("Done!")



# RUN MAIN FILE
if __name__ == "__main__":
    main()
