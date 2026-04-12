#!/bin/bash
# Download DNS Challenge 4 (ICASSP 2022) noisy speech dataset subset.

DEST=${1:-$HOME/datasets/dns}
mkdir -p $DEST
cd $DEST

git clone --depth 1 --filter=blob:none --sparse https://github.com/microsoft/DNS-Challenge.git
cd DNS-Challenge
git sparse-checkout set datasets/ICASSP_2022

grep -Eo 'https://[^ "]+noisy_fullband[^ "]+\.tar\.bz2' datasets/ICASSP_2022/download-dns-challenge-4.sh | head -5 | while read URL; do
    wget -P $DEST $URL
    tar xf $DEST/$(basename $URL) -C $DEST
    rm $DEST/$(basename $URL)
done
