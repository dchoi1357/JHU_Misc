#!/usr/bin/env bash

n=2
echo "==Tokenizing by removing punctuation and stop words=="
python parseCorpus.py ./data/cds14.txt ./data/cds14.inv ./data/cds14.dict $n
python cosineSim.py ./data/cds14.topics.txt ./data/cds14.inv \
	./data/cds14.dict $n ./wu-a.txt


n=3
echo ""
echo "==Tokenizing by removing punctuation and stop words also using 5-stem=="
python parseCorpus.py ./data/cds14.txt ./data/cds14_nstem.inv \
	./data/cds14_nstem.dict $n
python cosineSim.py ./data/cds14.topics.txt ./data/cds14_nstem.inv \
	./data/cds14_nstem.dict $n ./wu-b.txt
