GDURL="https://docs.google.com/uc?export=download"

downloadfile () {
    gid=$1
    fn=$2
    CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt \
            --keep-session-cookies --no-check-certificate \
            "$GDURL&id=$gid" -O- \
            | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')

    wget --load-cookies /tmp/cookies.txt \
        "${GDURL}&confirm=${CONFIRM}&id=$gid" \
        -O $fn && rm -rf /tmp/cookies.txt

    tar zxvf $fn

}

#base generators
#https://drive.google.com/file/d/1aLdcaO9J99UulzR9wozBh9d1LV878oLm/view?usp=sharing
BASEGEN_ID=1aLdcaO9J99UulzR9wozBh9d1LV878oLm
downloadfile $BASEGEN_ID base_generators.tar.gz

#samples
#https://drive.google.com/file/d/1qb-5BvlOTqstXn2yyNJwT7AqlUj2bkC2/view?usp=sharing
SAMPLES_ID=1qb-5BvlOTqstXn2yyNJwT7AqlUj2bkC2
downloadfile $BASEGEN_ID samples.tar.gz

#mr classifiers
#https://drive.google.com/file/d/1XTrVdVQ_vKrXEt83qf0m3NfVedRf1ifb/view?usp=sharing
CLASS_ID=1qb-5BvlOTqstXn2yyNJwT7AqlUj2bkC2
downloadfile $CLASS_ID base_classifiers.tar.gz

#aug generators
#https://drive.google.com/file/d/1hQpRR3VC1w8gNjlViOuLZGAukEHkR6Ij/view?usp=sharing
AUGGEN_ID=1hQpRR3VC1w8gNjlViOuLZGAukEHkR6Ij
downloadfile $AUGGEN_ID aug_generators.tar.gz
