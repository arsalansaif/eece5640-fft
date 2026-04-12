#!/bin/bash
# Download ORACLE RF dataset. Get download links from https://www.genesys-lab.org/oracle

DEST=${1:-$HOME/datasets/oracle}
mkdir -p $DEST

URLS=(
    # "https://example.com/WiFi_air_X310_3123D52_2ft_run1.tar.gz"
)

for URL in "${URLS[@]}"; do
    FNAME=$(basename $URL)
    wget -P $DEST $URL
    case $FNAME in
        *.tar.gz|*.tgz) tar xf $DEST/$FNAME -C $DEST && rm $DEST/$FNAME ;;
        *.zip)          unzip -q $DEST/$FNAME -d $DEST && rm $DEST/$FNAME ;;
    esac
done
