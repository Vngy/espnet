#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0


stage=1
stop_stage=1000
data_url=www.openslr.org/resources/12
train_set="train-clean-t"
train_dev="dev-clean-t"

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi
log "TORGO PARENT DIRECTORY: ${TORGO}"
if [ -z "${TORGO}" ]; then
    log "Fill the value of 'TORGO' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${TORGO}/torgo/LICENSE.TXT" ]; then
	echo "stage 1: Data Download to ${TORGO}"
	for part in dev-clean-t test-clean-t train-clean-t; do
            local/download_and_untar.sh ${TORGO} ${data_url} ${part}
	done
    else
        log "stage 1: ${TORGO}/torgo/LICENSE.TXT is already existing. Skip data downloading"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
	for part in dev-clean-t train-clean-t test-clean-t; do
        # use underscore-separated names in data directories.
        local/data_prep.sh ${TORGO}/torgo/${part} data/${part//-/_}
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: combine all training and development sets"
    utils/combine_data.sh --extra_files utt2num_frames data/${train_set} data/train_clean_t
    utils/combine_data.sh --extra_files utt2num_frames data/${train_dev} data/dev_clean_t
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # use external data
    if [ ! -e data/local/other_text/librispeech-lm-norm.txt.gz ]; then
	log "stage 4: prepare external text data from http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz"
        wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P data/local/other_text/
    fi
    if [ ! -e data/local/other_text/text ]; then
	# provide utterance id to each texts
	# e.g., librispeech_lng_00003686 A BANK CHECK
	zcat data/local/other_text/librispeech-lm-norm.txt.gz | \
	    awk '{ printf("librispeech_lng_%08d %s\n",NR,$0) } ' > data/local/other_text/text
    fi
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
