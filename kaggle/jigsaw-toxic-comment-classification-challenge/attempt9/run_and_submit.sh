#!/bin/sh

#time ipython get-vocab.py
#time ipython get-feat.py
time ipython train-pred.py $1
kaggle competitions submit -c jigsaw-toxic-comment-classification-challenge \
    -f ../data/attempt9/pred-9.csv -m "pred-9 C=$1"
