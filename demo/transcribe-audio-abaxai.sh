#!/bin/bash

if [ $# != 1 ]; then
    echo "Usage: transcribe-audio.sh <audio>"
    echo "e.g.: transcribe-audio.sh dr_strangelove.mp3"
    exit 1;
fi

! GST_PLUGIN_PATH=../src gst-inspect-1.0 kaldinnet2onlinedecoder > /dev/null 2>&1 && echo "Compile the plugin in ../src first" && exit 1;

if [ ! -f masterModel_juneASR/HCLG.fst ]; then
    echo "Run ./prepare-models.sh first to download models"
    exit 1;
fi

audio=$1

# --frames-per-chunk=51 --frame-subsampling-factor=3 --extra-left-context-initial=0
# --min-active=200 --max-active=7000 --beam=12.0 --lattice-beam=7.0 --acoustic-scale=1.0


GST_PLUGIN_PATH=../src gst-launch-1.0 --gst-debug="LIST" --gst-debug-level=2 -q filesrc location=$audio ! decodebin ! audioconvert ! audioresample ! \
kaldinnet2onlinedecoder \
  use-threaded-decoder=false \
  nnet-mode=3 \
  phone-syms=masterModel_juneASR/phones.txt \
  word-boundary-file=masterModel_juneASR/word_boundary.int \
  num-nbest=3 \
  num-phone-alignment=3 \
  do-phone-alignment=true \
  feature-type=mfcc \
  mfcc-config=conf/mfcc_abaxai.conf \
  ivector-extraction-config=conf/ivector_extractor.abaxai.conf \
  frames-per-chunk=51 \
  frame-subsampling-factor=3 \
  extra-left-context-initial=0 \
  min-active=200 \
  max-active=7000 \
  acoustic-scale=1.0 \
  beam=12.0 \
  lattice-beam=7.0 \
  do-endpointing=true \
  endpoint-silence-phones="1:2:3:4:5:6:7:8:9:10:11:12:13:14:15:16:17:18:19:20" \
  chunk-length-in-secs=0.25 \
  model=masterModel_juneASR/final.mdl \
  fst=masterModel_juneASR/HCLG.fst \
  word-syms=masterModel_juneASR/words.txt \
  hfst=masterModel_juneASR/HCLG.fst  \
  hword-syms=masterModel_juneASR/words.txt \
! filesink location=/dev/stdout buffer-mode=2

