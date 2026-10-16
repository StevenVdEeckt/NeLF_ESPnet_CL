import argparse
import numpy as np
import logging
import os
import shutil
import torch
import sentencepiece as spm
import torch.nn.functional as func
import inspect
import random
from collections import defaultdict

#######################################################################################################################
################################### HELP FUNCTIONS ####################################################################
#######################################################################################################################



#######################################################################################################################
################################### SAMPLING METHOD FUNCTIONS #########################################################
#######################################################################################################################

def uniform(utts, num_samples, probs=None):
    """
        Uniform sampling
    """
    return np.random.choice(utts, size=num_samples, replace=False, p=probs)


def speaker_uniform(utts, utt2spk, num_samples):
    """
        Speaker-uniform sampling
    
        Args:
            utts (list): List of utterance IDs.
            utt2spk (dict): Dictionary mapping utterance IDs to speaker IDs.
            num_samples (int): Number of utterances to sample.

        Returns:
            list: List of sampled utterance IDs.
    """
    # Group utterances by speakers
    spk2utts = defaultdict(list)
    for utt in utts:
        spk2utts[utt2spk[utt]].append(utt)

    speakers = list(spk2utts.keys())
    num_speakers = len(speakers)

    sampled_utts = []

    if num_samples <= num_speakers:
        # Sample num_samples speakers and one utterance per speaker
        sampled_speakers = random.sample(speakers, num_samples)
        for spk in sampled_speakers:
            sampled_utts.append(random.choice(spk2utts[spk]))
    else:
        # Sample floor(num_samples / num_speakers) utterances per speaker
        utts_per_speaker = num_samples // num_speakers
        remaining_samples = num_samples % num_speakers

        for spk in speakers:
            sampled_utts.extend(random.sample(spk2utts[spk], min(utts_per_speaker, len(spk2utts[spk]))))

        # Sample additional utterances from remaining speakers
        if remaining_samples > 0:
            sampled_speakers = random.sample(speakers, remaining_samples)
            for spk in sampled_speakers:
                if len(spk2utts[spk]) > utts_per_speaker:
                    sampled_utts.append(random.choice(spk2utts[spk]))

    return sampled_utts


def minimal_different_speakers(utts, utt2spk, num_samples):
    """
        minimal different Speakers sampling

        Args:
            utts (list): List of utterance IDs.
            utt2spk (dict): Dictionary mapping utterance IDs to speaker IDs.
            num_samples (int): Number of utterances to sample.

        Returns:
            list: List of sampled utterance IDs.
    """
    # Group utterances by speakers
    spk2utts = {}
    for utt in utts:
        if not utt2spk[utt] in spk2utts.keys():
            spk2utts[utt2spk[utt]] = []
        spk2utts[utt2spk[utt]].append(utt)

    speakers = list(spk2utts.keys())

    sampled_utts = []
    num_speakers = 0

    logging.info(f"Sampling {num_samples} utterances from {len(utts)} utterances and {len(speakers)} speakers..")

    while len(sampled_utts) < num_samples:
        sampled_speaker = random.sample(speakers, 1)[0]
        sampled_utts += spk2utts[sampled_speaker]
        speakers.remove(sampled_speaker)
        num_speakers += 1
        logging.info(f"num_speakers={num_speakers}, num_utterances={len(sampled_utts)} (goal: {num_samples})")

    return sampled_utts


def error_range_utterances(utts, utts2score, min_score, max_score, num_samples):
    utts2wer = torch.load(utts2score)
    assert min_score <= max_score
    utts_satisfying_range = [utt for utt in utts if min_score <= utts2wer[utt] <= max_score]
    logging.info(f"{len(utts_satisfying_range)} out of {len(utts)} (= {100.0 * len(utts_satisfying_range) / len(utts):.2f}%) utterances satisfy range [{min_score:.2f}, {max_score:.2f}]")
    if len(utts_satisfying_range) <= num_samples:
        return utts_satisfying_range
    return np.random.choice(utts_satisfying_range, size=num_samples, replace=False, p=None)


def uniform_per_task(utts, num_samples, utt2task, select_tasks=None):
    select_task = lambda task: (select_tasks is None or len(select_tasks) == 0) or (isinstance(select_tasks, list) and any([s in task for s in select_tasks]))
    task2utt = {}
    for utt, task in utt2task.items():
        if not select_task(task):
            continue
        if not task in task2utt.keys():
            task2utt[task] = [utt]
        else:
            task2utt[task].append(utt)
    logging.info(f"Tasks = {task2utt.keys()}")
    sampled_utts = None
    tot_num_samples = num_samples
    num_samples = int(num_samples / len(task2utt))
    # assign also the remaining samples
    rest_samples = tot_num_samples - num_samples * len(task2utt)
    N = len(task2utt)
    add_samples = [1] * rest_samples + [0] * (N - rest_samples)
    random.shuffle(add_samples)
    logging.info(f"Rest = {rest_samples}, to be divided as follows: {add_samples}")
    for i, (task, utts) in enumerate(task2utt.items(), 0):
        logging.info(f"Task {task} : sampling {min(num_samples + add_samples[i], len(utts))} utts out of {len(utts)}")
        s_utts = uniform(utts, num_samples + add_samples[i])
        if sampled_utts is None:
            sampled_utts = s_utts
        else:
            sampled_utts = np.concatenate((sampled_utts, s_utts))
    return sampled_utts


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
    parser.add_argument('--sampling_method', type=str, default='uniform', choices=['uniform', 'speaker', 'diff-speaker', 'error-range', 'uniform_per_task'], help="Sampling method, choose from ['uniform', 'speaker', 'diff-speaker', 'error-range']")
    parser.add_argument('--previous_memory', type=str, default='', help='To add a previous memory to the newly sampled memory')
    parser.add_argument('--score_file', type=str, default=None, help="Dictionary with utterances as keys and values as scores")
    parser.add_argument('--min_score', type=float, default=0, help="Minimal score from score_file for utterance to be a candidate for the sampled memory")
    parser.add_argument('--max_score', type=float, default=0, help="Maximal score from score_file for utterance to be a candidate for the sampled memory")
    parser.add_argument('--selected_tasks', type=str, default=None, help="Select only from given tasks")
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
    files = ['feats_type', 'spk2utt', 'text', 'utt2num_samples', 'utt2spk', 'wav.scp']

    # check if valid data dir
    for f in files:
        assert os.path.isfile("%s/%s" % (data_dir, f)), "file does not exist: %s/%s" % (data_dir, f)

    # some important things
    utt2spk, utt2txt, utt2txtprev, utt2txtctc = {}, {}, {}, {}
    utt2num_frames, utt2num_out = {}, {}
    utts = []
    utt2wav = {}

    # Reading all files
    logger.info("Reading files...")

    logger.info("...1/ '%s/utt2num_samples" % data_dir)

    with open('%s/utt2num_samples' % data_dir, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            utt, num_frames = line.strip("\n").split(" ")
            num_frames = int(num_frames)
            if num_frames > min_inputs:
                utt2num_frames[utt] = num_frames
            line = f.readline()

    logger.info("...2/ '%s/text" % data_dir)

    with open('%s/text' % data_dir, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            line = line.strip("\n").split(" ")
            utt, text = line[0], line[1:]
            if len(text) > min_outputs:
                utt2num_out[utt] = len(text)
                utt2txt[utt] = text
            line = f.readline()

    logger.info("...3/ '%s/utt2spk" % data_dir)

    with open('%s/utt2spk' % data_dir, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            utt, spk = line.strip("\n").split(" ")
            utt2spk[utt] = spk
            line = f.readline()

    with open('%s/wav.scp' % data_dir, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            utt, spk = line.strip("\n").split(" ")
            utt2wav[utt] = "/".join(spk.split("/")[:-3])
            line = f.readline()


    logger.info("Done! loading the files..")


    # computes intersection of lists
    def intersection(lst1, lst2):
        # lst3 = [value for value in lst1 if value in lst2]
        return list(set(lst1) & set(lst2))
        # return lst3

    # Check if utterances satisfy minimum input and output criteria
    utts = intersection(list(utt2num_frames.keys()), list(utt2num_out.keys()))
    logger.info("Utterances satisfying minimum input and output criteria: %d" % len(utts))

    # Sampling methods
    sampling_methods = {
        'uniform': uniform,
        'speaker': speaker_uniform,
        'diff-speaker': minimal_different_speakers,
        'error-range': error_range_utterances,
        'uniform_per_task': uniform_per_task,
    }

    # optional arguments for Sampling methods
    sm_args = {
        'utts': utts,
        'utt2txt': utt2txt,
        'utt2num_out': utt2num_out,
        'utt2num_frames': utt2num_frames,
        'utt2spk': utt2spk,
        'num_samples': num_samples,
        'utts2score': args.score_file,
        'min_score': args.min_score,
        'max_score': args.max_score,
        'utt2task': utt2wav,
        'select_tasks': args.selected_tasks.split(",") if args.selected_tasks is not None and len(args.selected_tasks) > 0 else args.selected_tasks,
    }

    # Start sampling
    logger.info("Sampling with sampling method.. = %s" % strategy)
    # get the arguments of the sampling method
    _args = {arg: val for arg, val in sm_args.items() if arg in inspect.getfullargspec(sampling_methods[strategy]).args}
    # run the sampling method
    selected_utts = sampling_methods[strategy](**_args)

    logger.info("Done! Memory contains %d utterances" % len(selected_utts))

    # will be needed later
    selected_spk = [utt2spk[utt] for utt in selected_utts]
    selected_num_frames = [utt2num_frames[utt] for utt in selected_utts]
    selected_num_out = [utt2num_out[utt] for utt in selected_utts]
    selected_txt = [utt2txt[utt] for utt in selected_utts]

    # saving the memory
    files_utt1 = ['text', 'utt2num_samples', 'utt2spk', 'wav.scp']
    files_copy = ['feats_type']
    files_spk1 = ['spk2utt']

    # naming
    min_len_tag = ""
    if min_inputs > 0:
        min_len_tag += "_i%d" % min_inputs
    if min_outputs > 0:
        min_len_tag += "_o%d" % min_outputs

    new_dir = "/".join(data_dir.split("/")[:-1]) + "/%s%s_%s_%d" % (args.name, min_len_tag, strategy, num_samples)

    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

    logger.info("Storing memory at %s.." % new_dir)

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

    # optional: add samples from a previous memory
    if old_data_dir:
        old_data_dir = old_data_dir.split(",")
        for old_data_diri in old_data_dir:
            logger.info("... adding samples from a previous memory: %s" % old_data_diri)
            for fname in files_utt1:
                fnew = open('%s/%s' % (new_dir, fname), 'a', encoding='utf-8')
                with open("%s/%s" % (old_data_diri, fname), 'r', encoding='utf-8') as fold:
                    line = fold.readline()
                    while line:
                        utt_ = line.split(" ")[0]
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

    # optional: add samples from a previous memory
    if old_data_dir:
        for old_data_diri in old_data_dir:
            for fname in files_spk1:
                fnew = open('%s/%s' % (new_dir, fname), 'a', encoding='utf-8')
                with open("%s/%s" % (old_data_diri, fname), 'r', encoding='utf-8') as fold:
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
