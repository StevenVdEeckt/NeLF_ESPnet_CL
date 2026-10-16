#!/usr/bin/env bash

_nj=$1
train_cmd=$2
featsdir=$3

. ./path-kaldi-singularity.sh

steps/make_fbank_pitch.sh --nj "${_nj}" --cmd "${train_cmd}" "${featsdir}"
