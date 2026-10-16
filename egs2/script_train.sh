scl enable devtoolset-7 bash &&
export PATH=/users/spraak/spch/prog/spch/cuda-11.0/bin${PATH:+:${PATH}} &&
export LD_LIBRARY_PATH=/users/spraak/spch/prog/spch/cuda-11.0/lib64:/users/spraak/spch/prog/spch/cudnn-7.6/lib64:/.singularity.d/libs &&
export CFLAGS="-I/users/spraak/spch/prog/spch/cuda-11.0/include $CFLAGS" &&
export CUDA_HOME=/users/spraak/spch/prog/spch/cuda-11.0 &&
export CUDA_PATH=/users/spraak/spch/prog/spch/cuda-11.0 &&
export KALDI_ROOT=/esat/spchtemp/spchdisk_orig/svandere/espnet/nelf_espnet2/NeLF_Transcription_ASR/tools/kaldi
export PATH=$KALDI_ROOT/egs/wsj/s5/utils:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/bin:$KALDI_ROOT/src/featbin:$KALDI_ROOT/src/lmbin:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/fstbin:$KALDI_ROOT/src/gmmbin:$KALDI_ROOT/src/latbin:$PATH

source /esat/spchtemp/spchdisk_orig/svandere/espnet/espnet3/espnet/tools/venv/envs/esp3/etc/profile.d/conda.sh
conda activate nelf_env

which python
which run.pl
#python -m espnet2.bin.launch --help | head

#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# 1. SETTING DEFAULT VALUES FOR ARGUMENTS

# set to 0 to use cpu
ngpu=1

# the stage to run
stage=11
stop_stage=11

# some parameters related to model(name)
exp=exp/nelf              # directory of the models
asr_suffix=             # suffix of the model directory
nbpe=20000              # number of outputs of model
pretrain=false          # set to True if pretrained model should be loaded
token_type=bpe
asr_stats_suffix=
token_lang=             # (optionally) language of the token list
token_mask=
cleaner=none

# some parameters related to pretrained model(name)
pt_arch=                # architecture (i.e. config file) of the pretrained model
pt_tag=                 # (optionally) the tag of the pretrained model
pt_suffix=              # (optionally) the suffix of the pretrained model

# some parameters related to memory
mem_tag=                # (optionally) tag of the memory

#token_mask=mask_clm_bpe_unigram${nbpe}.npy
num_jobs=1

# extra params
model_name=valid.acc.ave.pth            # name of the model (to decode)
asr_config=                             # configuration of the model
inference_config=conf/decode_asr.yaml   # inference configuration
old_lang=                               # previous task (optional)
init_param=
data_dir=
feats_normalize=global_mvn
transcripts=text                        # name of the transcripts file, by default .../text
transfer_org_sets=
memory_set=
transfer_data=true


# 2. SET PARAMETERS BASED ON $1 VALUE
case "$1" in

       0)    
    asr_config=conf/continual_learning/train_nelf_svr_20.yaml
    task=chorec_avi_word
    nbpe=5000
    pretrain=true
    pretrained_model=st_train_nelf_finetuning_raw_chorec_avi_word_bpe_lc5000
    model_name=latest.pth
    token_lang=nelf
    train_set=CHOREC/train_avi_word
    valid_set=CHOREC/dev_avi_word
    test_sets=CHOREC/test
    memory_set=CGN/memory_uniform_20
    mem_tag=uniform_20
    ;;


       1)
    asr_config=conf/continual_learning/train_nelf_svr_20_encoder.yaml
    task=chorec_avi_word
    nbpe=5000
    pretrain=true
    pretrained_model=st_train_nelf_finetuning_encoder_raw_chorec_avi_word_bpe_lc5000
    model_name=latest.pth
    token_lang=nelf
    train_set=CHOREC/train_avi_word
    valid_set=CHOREC/dev_avi_word
    test_sets=CHOREC/test
    memory_set=CGN/memory_uniform_20
    mem_tag=uniform_20
    ;;


       2)
    asr_config=conf/continual_learning/train_nelf_svr_20_encoder6.yaml
    task=chorec_avi_word
    nbpe=5000
    pretrain=true
    pretrained_model=st_train_nelf_finetuning_encoder6_raw_chorec_avi_word_bpe_lc5000
    model_name=latest.pth
    token_lang=nelf
    train_set=CHOREC/train_avi_word
    valid_set=CHOREC/dev_avi_word
    test_sets=CHOREC/test
    memory_set=CGN/memory_uniform_20
    mem_tag=uniform_20
    ;;


    #   3)
    #asr_config=conf/continual_learning/train_nelf_svr_20_encoder6.yaml
    #task=chorec_avi_word
    #nbpe=5000
    #pretrain=true
    #pretrained_model=st_train_nelf_finetuning_encoder6_raw_chorec_avi_word_bpe_lc5000
    #model_name=latest.pth
    #token_lang=nelf
    #train_set=CHOREC/train
    #valid_set=CHOREC/dev
    #test_sets=CHOREC/test
    #memory_set=CGN/memory_uniform_20
    #mem_tag=uniform_20
    #;;



























    	*)
    echo "Unsupported configuration: $1"
    exit 1
        ;;
esac




# Set the dataset based on the task
#train_set="${data_dir}train" #_${task}"
#valid_set="${data_dir}dev" #_${task}"
#test_sets="${data_dir}test" #_${task}"
# Set the language (i.e. task)
lang=${task}

# Set the memory tag
if [ -z ${memory_set} ]; then
    memory_mode=
else
    memory_mode="memory_${mem_tag}"
fi

# Add pt_tag or not
if [[ ! -z ${pt_tag} ]]; then
    pt_tag=_${pt_tag}
fi

# Add pt_suffix or not
if [ ! -z ${pt_suffix} ]; then
   pt_suffix=_${pt_suffix}
fi

# Model is pre-trained or not
if [[ "${pretrain}" = false ]]; then
    pretrained_model=""
else
    pretrained_model=${exp}/${pretrained_model}/${model_name}
    if [[ ! -z ${init_param} ]]; then
        pretrained_model=${pretrained_model}:${init_param}
    fi
fi


# Define your arguments here as a single string
arguments=(
    --ngpu "$ngpu" 
    --nj "${num_jobs}" 
    --stage "${stage}" 
    --stop_stage "${stop_stage}"
    --expdir "${exp}"
    --st_config "${asr_config}"
    --feats_normalize "${feats_normalize}"
    --use_lm "false"
    --token_lang "${token_lang}"
    --use_ngram "false" 
    --pretrained_asr "${pretrained_model}"
    --lang "${lang}"
    --inference_config "${inference_config}"
    --st_train_set "${train_set}"
    --st_valid_set "${valid_set}"
    --st_test_set "${test_sets}"
    --memory_set "${memory_set}"
    --memory_mode "${memory_mode}"
    --copy_files "${transfer_data}"
    --src_lang "verbatim"
    --tgt_lang "subtitle"
    --src_token_type "bpe" 
    --src_nbpe 5000 
    --src_bpemode "unigram"
    --src_case "lc"  
    --tgt_token_type "bpe" 
    --tgt_nbpe 5000 
    --tgt_bpemode "unigram"
    --tgt_case "lc"  
    --oov "<unk>" 
    --token_joint true
    --audio_format "ark"
    --feats_type "raw"
    --feats_normalize "global_mvn"
)

script_name="./subs_st1.sh"

# Run the chosen script with the arguments
"$script_name" "${arguments[@]}"

