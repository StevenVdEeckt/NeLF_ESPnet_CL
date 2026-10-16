#!/usr/bin/env bash
set -euo pipefail

###### TRANSFERING TO LOCAL MACHINE ##################
# Script to transfer data from audioslave to local machine
# The script make the following assumptions:
#  - the data is stored on audioslave 
#  - the paths in the ${scp} file point to ${data_feats}/${train_set}
#     (i.e. not to ${data_feats}/org/${train_set} as ESPnet usually does
echo "[INFO] Transfering files to $_CONDOR_SCRATCH_DIR" >&2

# set params
# ----- default valus ----
current_dir=
train_set=
valid_set=
data_feats="dump/raw"
scp="wav.scp"

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --current_dir) current_dir="$2"; shift 2;;
        --train_set) train_set="$2"; shift 2;;
        --valid_set) valid_set="$2"; shift 2;;
        --data_feats) data_feats="$2"; shift 2;;
        --scp) scp="$2"; shift 2;;
        *) echo "[ERROR] Unknown argument: $1" >&2; exit 1;;
    esac
done

# ---- Safety checks ----
if [[ -z "$current_dir" || -z "$train_set" ]]; then
    echo "[ERROR] Missing required args (--current_dir, --train_set)" >&2
    exit 1
fi


# data_dir is same as data_feats
data_dir=${data_feats}

# set dir_id  based on _CONDOR_SCRATCH_DIR
dir_id=${_CONDOR_SCRATCH_DIR#/usr/data/condor/execute/}
dir_id=${dir_id#/volume1/condor/execute/}

# we will transfer over SSH
# Extract username from the path (the element after '/audioslave/')
username=$(echo "$current_dir" | awk -F'/' '{for(i=1;i<=NF;i++) if($i=="audioslave") print $(i+1)}')
# Replace the prefix '/esat/audioslave/<user>' with 'audioslave:/usr/data/<user>'
audioslave=$(echo "$current_dir" | sed -E "s|^/esat/audioslave/${username}|audioslave:/usr/data/${username}|")

# 0. log in on ssh  
eval "$(ssh-agent -s)" >/dev/null
ssh-add ~/.ssh/id_rsa

# 1. transfer dump/raw files to local machine 
# note: in case ${train_set} or ${valid_set} are itself paths, the path first has to be made
mkdir -p "$_CONDOR_SCRATCH_DIR/${train_set}"
if [ ! -z "${valid_set}" ]; then
    mkdir -p "$_CONDOR_SCRATCH_DIR/${valid_set}"
fi
# now the data can be transfered
rsync -aLv "${audioslave}/${data_feats}/${train_set}/" "$_CONDOR_SCRATCH_DIR/${train_set}" >&2
if [ ! -z "${valid_set}" ]; then
    rsync -aLv "${audioslave}/${data_feats}/${valid_set}/" "$_CONDOR_SCRATCH_DIR/${valid_set}" >&2
fi

# some checks
echo "[CHECK] ls -l $_CONDOR_SCRATCH_DIR" >&2
ls -l $_CONDOR_SCRATCH_DIR >&2
echo "[CHECK] ls -l $_CONDOR_SCRATCH_DIR/${train_set}" >&2
ls -l $_CONDOR_SCRATCH_DIR/${train_set} >&2
if [ ! -z "${valid_set}" ]; then
   echo "[CHECK] ls -l $_CONDOR_SCRATCH_DIR/${valid_set}" >&2
   ls -l $_CONDOR_SCRATCH_DIR/${valid_set} >&2
fi

# 2. close ssh connection
ssh-agent -k >/dev/null

# 3. In wav.scp from 1., set links to local machine
#
# wav.scp may contain paths using either:
#
#   /esat/audioslave/<user>/...
#
# or:
#
#   /esat/spchtemp/spchdisk_orig/<user>/...
#
# The actual files have already been copied to:
#
#   $_CONDOR_SCRATCH_DIR
#
# Therefore replace either possible original root by the scratch directory.


# current_dir, for example:
# /esat/audioslave/svandere/espnet/asr/nelf
#
# Construct the corresponding spchtemp path:
# /esat/spchtemp/spchdisk_orig/svandere/espnet/asr/nelf

spchtemp_current_dir=$(echo "$current_dir" | \
    sed -E "s|^/esat/audioslave/${username}|/esat/spchtemp/spchdisk_orig/${username}|")


rewrite_wav_scp () {
    local scp_file="$1"

    # Both possible roots that may occur in wav.scp
    local old_roots=(
        "${current_dir}/${data_dir}"
        "${spchtemp_current_dir}/${data_dir}"
    )

    echo "[INFO] Rewriting paths in ${scp_file}" >&2

    for old_root in "${old_roots[@]}"; do

        if grep -Fq "$old_root" "$scp_file"; then
            echo "[INFO] Found path prefix:" >&2
            echo "[INFO]   ${old_root}" >&2
            echo "[INFO] Replacing with:" >&2
            echo "[INFO]   ${_CONDOR_SCRATCH_DIR}" >&2

            sed -i \
                "s|${old_root}|${_CONDOR_SCRATCH_DIR}|g" \
                "$scp_file"
        fi
    done
}


# Rewrite train wav.scp
rewrite_wav_scp \
    "$_CONDOR_SCRATCH_DIR/${train_set}/${scp}"


# Rewrite validation wav.scp
if [[ -n "${valid_set}" ]]; then
    rewrite_wav_scp \
        "$_CONDOR_SCRATCH_DIR/${valid_set}/${scp}"
fi


# Safety check:
# after rewriting, there should be no /esat/... ARK paths left.

if grep -Fq "/esat/" \
    "$_CONDOR_SCRATCH_DIR/${train_set}/${scp}"; then

    echo "[ERROR] Some paths in train wav.scp were not rewritten:" >&2
    grep -F "/esat/" \
        "$_CONDOR_SCRATCH_DIR/${train_set}/${scp}" \
        | head -n 10 >&2

    exit 1
fi


if [[ -n "${valid_set}" ]]; then

    if grep -Fq "/esat/" \
        "$_CONDOR_SCRATCH_DIR/${valid_set}/${scp}"; then

        echo "[ERROR] Some paths in valid wav.scp were not rewritten:" >&2
        grep -F "/esat/" \
            "$_CONDOR_SCRATCH_DIR/${valid_set}/${scp}" \
            | head -n 10 >&2

        exit 1
    fi
fi


# Check if rewriting was done properly
echo "[INFO] First lines of $_CONDOR_SCRATCH_DIR/${train_set}/${scp}" >&2
head -n 10 \
    "$_CONDOR_SCRATCH_DIR/${train_set}/${scp}" >&2

if [[ -n "${valid_set}" ]]; then
    echo "[INFO] First lines of $_CONDOR_SCRATCH_DIR/${valid_set}/${scp}" >&2
    head -n 10 \
        "$_CONDOR_SCRATCH_DIR/${valid_set}/${scp}" >&2
fi

# OLD VERSION
# Define the old string to be replaced
#old_string="${current_dir}/${data_dir}"

# Use the old_string variable in the sed command, making sure to escape slashes
#escaped_old_string=$(echo "$old_string" | sed 's_/_\\/_g')

# print command
#echo "[CHECK] sed -i \"s/${escaped_old_string}/${_CONDOR_SCRATCH_DIR//\//\\/}/g\" $_CONDOR_SCRATCH_DIR/${train_set}/${scp}" >&2

# for train and valid set at level dump/raw
#sed -i "s/${escaped_old_string}/${_CONDOR_SCRATCH_DIR//\//\\/}/g" "$_CONDOR_SCRATCH_DIR/${train_set}/${scp}"
#if [ ! -z "${valid_set}" ]; then
#    sed -i "s/${escaped_old_string}/${_CONDOR_SCRATCH_DIR//\//\\/}/g" "$_CONDOR_SCRATCH_DIR/${valid_set}/${scp}"
#fi

# check if done properly
#echo "[INFO] First lines of $_CONDOR_SCRATCH_DIR/${train_set}/${scp}" >&2
#head -n 10 $_CONDOR_SCRATCH_DIR/${train_set}/${scp} >&2
#if [ ! -z "${valid_set}" ]; then
#    echo "[INFO] First lines of $_CONDOR_SCRATCH_DIR/${valid_set}/${scp}" >&2
#    head -n 10 $_CONDOR_SCRATCH_DIR/${valid_set}/${scp} >&2
#fi

# 4. update the train and valid dirs so that they point to symlink
_asr_train_dir="$_CONDOR_SCRATCH_DIR/${train_set}"
if [ ! -z "${valid_set}" ]; then
   _asr_valid_dir="$_CONDOR_SCRATCH_DIR/${valid_set}"   
fi

if [ ! -z "${valid_set}" ]; then
   echo "${_asr_train_dir} ${_asr_valid_dir}"
else
   echo "${_asr_train_dir}"
fi

###### END OF TRANSFER TO LOCAL MACHINE ##################
