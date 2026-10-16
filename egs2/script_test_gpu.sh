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

export KALDI_ROOT=/esat/spchtemp/spchdisk_orig/svandere/espnet/nelf_espnet2/NeLF_Transcription_ASR/tools/kaldi
export PATH=$KALDI_ROOT/egs/wsj/s5/utils:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/bin:$KALDI_ROOT/src/featbin:$KALDI_ROOT/src/lmbin:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/fstbin:$KALDI_ROOT/src/gmmbin:$KALDI_ROOT/src/latbin:$PATH

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
stage=12
stop_stage=13

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
old_lang=                               # previous task (optional)
init_param=
data_dir=
feats_normalize=global_mvn
transcripts=text                        # name of the transcripts file, by default .../text
transfer_org_sets=
memory_set=
transfer_data=true

train_set="train"
valid_set="dev"
test_dir=""

token_lang=nelf

test_sets=(
"FIIAT/MarieK_WEAVE_Juli26"
#"CHOREC/train_avi_uniform_100"
#"CHOREC/dev_avi"
#"CHOREC/test_avi"
#"CHOREC/test_pseudo"
#"CHOREC/test_word"
#"SABeD/test"
)

test_langs=(
"nl"
"nl"
"nl"
"nl"
"nl"
"nl"
)

inference_config=conf/decode_verbatim.yaml

getmodelconfig() {
    local model=$1
    local test_lang=$2
    case $model in

        0)      
                transfer_data=false
		asr_config=conf/continual_learning/train_nelf_finetuning.yaml
                task=chorec
                nbpe=5000
                inference_asr_model=init_model.pth
        ;;

        1)
                asr_config=conf/continual_learning/train_nelf_finetuning_encoder.yaml
                task=chorec_avi_word
                nbpe=5000
                inference_asr_model=latest.pth
        ;;

        2)
                asr_config=conf/continual_learning/train_nelf_finetuning_encoder6.yaml
                task=chorec_avi_word
                nbpe=5000
                inference_asr_model=latest.pth
        ;;

        3)
                asr_config=conf/continual_learning/train_nelf_finetuning.yaml
                task=chorec_avi_word
                nbpe=5000
                inference_asr_model=avg_model.pth
        ;;

        4)
                asr_config=conf/continual_learning/train_nelf_finetuning_encoder.yaml
                task=chorec_avi_word
                nbpe=5000
                inference_asr_model=avg_model.pth
        ;;

        5)
                asr_config=conf/continual_learning/train_nelf_finetuning_encoder6.yaml
                task=chorec_avi_word
                nbpe=5000
                inference_asr_model=avg_model.pth
        ;;

        6)
                asr_config=conf/continual_learning/train_nelf_svr_20.yaml
                task=chorec_avi_word
                nbpe=5000
                inference_asr_model=latest.pth
        ;;

        7)
                asr_config=conf/continual_learning/train_nelf_svr_20_encoder.yaml
                task=chorec_avi_word
                nbpe=5000
                inference_asr_model=latest.pth
        ;;

        8)
                asr_config=conf/continual_learning/train_nelf_svr_20_encoder6.yaml
                task=chorec_avi_word
                nbpe=5000
                inference_asr_model=latest.pth
        ;;


        # Add more models as needed
        *)
            echo "Unsupported model configuration: $model"
            exit 1

        	;;
  esac

}

# GET CONFIG
# Input parameter
input_parameter=$1

# Calculate model index and test set index
model_index=$((input_parameter / ${#test_sets[@]}))
test_set_index=$((input_parameter % ${#test_sets[@]}))

# Extract the selected language
test_lang=${test_langs[$test_set_index]}

# Load model configuration with language consideration
getmodelconfig $model_index $test_lang

# Set the language (i.e. task)
lang=${task}

# Extract the selected test set
test_set=${test_dir}${test_sets[$test_set_index]}

echo "asr_config=${asr_config}"
echo "lang=${lang}"
echo "test_sets=${test_sets}"
echo "test_set=${test_set}"
echo "inference_asr_model=${inference_asr_model}"


# Define your arguments here as a single string
arguments=(
    --ngpu "$ngpu" 
    --nj "${num_jobs}" 
    --stage "${stage}"
    --copy_files "${transfer_data}" 
    --stop_stage "${stop_stage}"
    --expdir "${exp}"
    --st_config "${asr_config}"
    --feats_normalize "${feats_normalize}"
    --use_lm "false"
    --token_lang "${token_lang}"
    --use_ngram "false" 
    --lang "${lang}"
    --inference_config "${inference_config}"
    --st_train_set "${train_set}"
    --st_valid_set "${valid_set}"
    --st_test_set "${test_set}"
    --inference_nj "${num_jobs}"
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
    --inference_st_model "${inference_asr_model}"
)

script_name="./subs_st1.sh"

# Run the chosen script with the arguments
"$script_name" "${arguments[@]}"

