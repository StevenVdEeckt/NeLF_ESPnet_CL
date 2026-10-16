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
import subprocess
import datetime, string

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


def uniform_per_task(utts, num_samples, utt2task):
    task2utt = {}
    for utt, task in utt2task.items():
        if not task in task2utt.keys():
            task2utt[task] = [utt]
        else:
            task2utt[task].append(utt)
    sampled_utts = None
    logging.info(f"There are {len(task2utt)} tasks to sample from!")
    for task, utts in task2utt.items():
        logging.info(f"Task {task} : sampling {min(num_samples, len(utts))} utts out of {len(utts)}")
        s_utts = uniform(utts, num_samples)
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
    files = ['feats_type', 'spk2utt', 'text', 'text.lc.verbatim', 'text.lc.subtitle', 'utt2num_samples', 'utt2spk', 'wav.scp']

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
            l = line.strip("\n").split(" ") 
            utt, spk = line.strip("\n").split(" ")
            utt2wav[utt] = "/".join(spk.split("/")[:-3])
            line = f.readline()

    # generate task label or scan old task labels
    utt2task = {}
    if os.path.isfile(f"{data_dir}/utt2task"):
        logger.info("Yes, I found an utt2task file")
        with open(f"{data_dir}/utt2task", "r", encoding="utf-8") as fold:
            line = fold.readline()
            while line:
                utt, task = line.strip("\n").split(" ")
                utt2task[utt] = task
                line = fold.readline() 
    else:
        # generate task label
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        rand_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        task_label = f"task_{timestamp}_{rand_suffix}"
        utt2task = {utt: task_label for utt in utt2spk.keys()}

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
        'utt2task': utt2task,
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
    files_utt1 = ['text', 'utt2num_samples', 'utt2spk', 'wav.scp', 'text.lc.subtitle', 'text.lc.verbatim']
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
        logger.info("... adding samples from a previous memory: %s" % old_data_dir)
        for fname in files_utt1:
            fnew = open('%s/%s' % (new_dir, fname), 'a', encoding='utf-8')
            with open("%s/%s" % (old_data_dir, fname), 'r', encoding='utf-8') as fold:
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
        for fname in files_spk1:
            fnew = open('%s/%s' % (new_dir, fname), 'a', encoding='utf-8')
            with open("%s/%s" % (old_data_dir, fname), 'r', encoding='utf-8') as fold:
                line = fold.readline()
                while line:
                    spk_ = line.split(" ")[0]
                    fnew.write(line)
                    line = fold.readline()
            fnew.close()

    # finally: generate a utt2task file
    with open("%s/utt2task" % new_dir, "w", encoding="utf-8") as fnew:
        for utt in selected_utts:
            fnew.write(f"{utt} {utt2task[utt]}\n")
    # optional: add samples from a previous memory
    if old_data_dir:
       lines_to_add = []
       # if utt2task file exists, use its lines to copy to utt2task
       if os.path.isfile("%s/utt2task" % old_data_dir):
           with open("%s/utt2task" % old_data_dir, "r", encoding="utf-8") as fold:
               line = fold.readline()
               while line:
                   lines_to_add.append(line)
                   line = fold.readline()
       else:
           # if utt2task does not exist, scan utterances and generate task id
           # generate old task label
           timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
           rand_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
           old_task_label = f"task_{timestamp}_{rand_suffix}"
           with open("%s/text" % old_data_dir, "r", encoding="utf-8") as fold:
                line = fold.readline()
                while line:
                    utt = line.strip("\n").split(" ")[0]
                    lines_to_add.append(f"{utt} {old_task_label}\n")
                    line = fold.readline()
       # write to utt2task
       with open("%s/utt2task" % new_dir, "a", encoding="utf-8") as fnew:
           for line in lines_to_add:
               fnew.write(line)

    logger.info("Done!")

    # finally: transfer data files
    # step 1: make dir {data_dir}/data
    data_subdir = os.path.join(new_dir, "data")
    os.makedirs(data_subdir, exist_ok=True)
    # step 2: get utts and wav files
    utt2wav, utt2new_wav = {}, {}
    with open(f"{new_dir}/wav.scp", "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            utt, wav_file = line.strip().split(maxsplit=1)
            utt2wav[utt] = wav_file
            line = f.readline()
    # step 3: make symlink from file to {data_dir}/data
    for utt, wav_file in utt2wav.items():
        task_name = utt2task[utt]
        # get wav file path '/format.1/fn000005_N00059_000367004_000395248_nl_asr.flac' from 
        # path: /esat/audioslave/svandere/espnet/asr/ml/dump/raw/owsm/cgn/train_cgnnl/data/format.1/fn000005_N00059_000367004_000395248_nl_asr.flac
        ark_file = wav_file.split(":", 1)[0]
        ark_number = wav_file.split(":", 1)[1]
        # 1. Extract path after "data/"
        rel_path = ark_file.split("/data/", 1)[1]
        # 2. Prepend the task name
        rel_path = os.path.join(task_name, rel_path)
        # 3. Compute full target path inside the memory dataset
        link_path = os.path.join(data_subdir, rel_path)
        # 4. Ensure all intermediate directories exist
        os.makedirs(os.path.dirname(link_path), exist_ok=True)
        # run bash command: "ln -s {wav_file} {data_dir}/data/{link_path}" 
        if not os.path.islink(link_path):
            try:
                os.symlink(ark_file, link_path)
            except FileExistsError:
                pass  # already linked, fine
        # so "store" file in {data_dir}/data through a symlink
        # update wav file path in wav.scp to $(pwd)/${data_dir}/data/${wav_file_name}
        new_wav_file = os.path.abspath(link_path)
        utt2new_wav[utt] = str(new_wav_file) + f":{ark_number}"
    # step 4: store new wav paths
    with open(f"{new_dir}/wav.scp", "w", encoding="utf-8") as f:
        for utt, wav_file in utt2new_wav.items():
            f.write(f"{utt} {wav_file}\n")
    # Done



# RUN MAIN FILE
if __name__ == "__main__":
    main()
