#!/usr/bin/env python
# coding: utf-8


# do all the imports
import numpy as np
import pandas as pd
import normalization.score as dtw
import argparse
import logging
import re
import torch

#import sys
#sys.path.append('/esat/spchtemp/spchdisk_orig/svandere/espnet/asr/ml')


FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT,  level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger('main')

def strtobool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

""" Parsing command line arguments..."""
parser = argparse.ArgumentParser(description='Scoring with normalization and compounding...')
parser.add_argument('decode_dir',  type=str, help='Path to decode directory')
args = parser.parse_args()

""" Setting the reference (ref) and hypothesis (hyp) files... """
decode_dir = args.decode_dir
ref = decode_dir + "/ref.trn"
hyp = decode_dir + "/hyp.trn" 

logger.info('Scoring: %s..' % decode_dir)

""" Function required for score """
split = lambda x: x[:x.find('(')]
get_name = lambda x: x[x.find('(')+1:-1]

""" Remove functional tokens """
def remove_functional_tokens(text):
    # Regular expression to match "<...>" where ... does not include ">"
    pattern = r'<[^>]*>'
    # Replace occurrences of the pattern with an empty string
    clean_text = re.sub(pattern, '', text)
    return clean_text


""" Own Tokenizer: for all punctuation must be removed """
def tokenizer(text):
    for symbol in ['"', ".", ":", ";", ",", "!", "?", "<unk>", "<nospeech>", " X "]:
        text = text.replace(symbol, "")
    return remove_functional_tokens(text).lower()


def score_one(ref, hyp):
    dtwdist, alignment, results = dtw.dtw(dtw.tokenizer(hyp), dtw.tokenizer(ref), Compounds=True,Dash=True, Verbose=False)
    return 100.0 * (results[0] + results[1] + results[2]) / results[3]


"""
    Score the corpus - can be used to compute the CER
    :param str ref: reference file name
    :param str hyp: hypothesis file name
    :param int k: (optional) only utterances with at least k output symbols are considered, by default k = 1
    :param bool cer: (optional) indicate if CER must be computed
"""
def score(ref, hyp, k=2):
    scores = {}
    f_hyp = open(hyp, "r")
    with open(ref, "r") as f_ref:
        line_r = f_ref.readline()
        line_h = f_hyp.readline()
        while line_r and line_h: 
            # get utterance name
            utt_id = get_name(line_r)
            # prepare utterance
            line_r = split(line_r)
            line_h = split(line_h).replace('<sos/eos>', '')            
            line_r = tokenizer(line_r)
            line_h = tokenizer(line_h)
            # score
            s = score_one(line_r, line_h)
            # store score
            scores[utt_id] = s
            num_utts = len(scores)
            logging.info(f"Processed {' ' * (7 - len(str(len(num_utts))))}{num_utts}.. WER = {s:.2f}")
            # move onto next line
            line_r = f_ref.readline()
            line_h = f_hyp.readline()
    f_hyp.close()
    return scores

""" We score the corpora for all four settings... """
scores = score(ref, hyp)
store_file = "/".join(decode_dir.split("/")[:-1]) + "/scores_wer.pth"
torch.save(scores, score_file) 
logging.info("Stored WER-scores at: {store_file}")
logger.info("-" * 100)
