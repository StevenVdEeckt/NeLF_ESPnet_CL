#!/usr/bin/env bash
export PYTHONPATH=""
source /esat/audioslave/pmehrali/anaconda3/bin/activate nelf_sm120
# source /esat/audioslave/pmehrali/anaconda3/bin/activate nelf_sm120
#source /esat/spchtemp/spchdisk_orig/pmehrali/espnet/tools/miniconda/envs/espnet/bin/activate
#source /esat/spchtemp/scratch/jponcele/anaconda3/bin/activate espnet21

python --version

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

####################################################
stage=11
stop_stage=11
####################################################

# notes: geen speed perturbation, geen LM, geen word LM, geen NGRAM LM

# Always enable copying of data to local worker node for better performance
copy_files=true
data_machine="audioslave"  # Source machine where data is stored

nj=8
ngpu=1

# # DATA
feats_type=fbank_pitch
audio_format=wav


st_train_set="Train"
st_valid_set="Val"
st_test_set="Test"

#####################################################################################
# EXP
outdir=/esat/audioslave/pmehrali/experiments/GCND/long_models/subtitle_long_unk
expdir=${outdir}/exp
st_tag=Subtitle_long_unk_new_try


dumpdir=/path/to/dump 

pretrained_model=/path/to/nelf/pretrained/model.pth

st_config=path/to/train.yaml
#####################################################################################


# LM
use_word_lm=false  # not yet supported!
use_lm=false
lm_config=conf/train_lm_transformer.yaml
use_ngram=false

# ST
feats_normalize=utterance_mvn  # recommended for pretrained models instead of globalmvn

inference_config=conf/st_decode_chained.yaml  #_verbatim_only.yaml
inference_nj=8  #8  #64
gpu_inference=true #false
#st_args="--batch_type custom_folded --valid_batch_type custom_folded"  # "--input_size 0"  # to use raw audio for w2v2 encoder

st_args="--batch_type custom_folded --valid_batch_type custom_folded --batch_asr_ratio 0.0" # 1.0 for verbatim only and 0.0 for subtitle only
inference_st_model=valid.acc_subs.ave_10best.pth #valid.acc_asr.best.pth  #averaged_model.pth  #valid.acc_asr.ave.pth

echo "Starting training with dumpdir=${dumpdir} and copy_files=${copy_files}"

./subs_st1.sh \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --ngpu ${ngpu}  \
    --nj ${nj}  \
    --gpu_inference ${gpu_inference}  \
    --dumpdir ${dumpdir} \
    --expdir ${expdir}  \
    --feats_type ${feats_type}  \
    --audio_format wav  \
    --min_wav_duration 0.1  \
    --max_wav_duration 30  \
    --token_joint true \
    --src_token_type bpe \
    --src_nbpe 5000 \
    --src_bpemode unigram \
    --src_case lc  \
    --tgt_token_type bpe \
    --tgt_nbpe 5000 \
    --tgt_bpemode unigram \
    --tgt_case lc  \
    --oov "<unk>" \
    --lang "vl" \
    --src_lang "verbatim" \
    --tgt_lang "subtitle" \
    --use_lm ${use_lm} \
    --use_word_lm ${use_word_lm}  \
    --lm_config ${lm_config}  \
    --use_ngram ${use_ngram}  \
    --st_args "${st_args}"  \
    --st_tag ${st_tag}  \
    --inference_config ${inference_config}  \
    --inference_nj ${inference_nj}  \
    --feats_normalize ${feats_normalize}  \
    --st_train_set "${st_train_set}" \
    --st_valid_set "${st_valid_set}" \
    --st_test_set "${st_test_set}" \
    --inference_st_model ${inference_st_model}  \
    --gpu_inference ${gpu_inference} \
    --overwrite_bpemodel /esat/spchtemp/spchdisk_orig/pmehrali/NeLF_Transcription_ASR/egs2/cgn/nelf_transcribe/model/ASR_subtitles_v2/bpe.model \
    --overwrite_token_list /esat/spchtemp/spchdisk_orig/pmehrali/NeLF_Transcription_ASR/egs2/cgn/nelf_transcribe/model/ASR_subtitles_v2/tokens.txt \
    --pretrained_asr ${pretrained_model} \
    --st_config ${st_config}  \
    --copy_files ${copy_files} \
    --data_machine ${data_machine} \



