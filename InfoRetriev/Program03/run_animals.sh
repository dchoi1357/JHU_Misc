#!/usr/bin/env bash
if [ -z $1 ] ; then
	n=0
else
	n=$1
fi

python parseCorpus.py ./data/animal.txt ./data/animal.inv ./data/animal.dict $n
python cosineSim.py ./data/animal.topics.txt ./data/animal.inv \
	./data/animal.dict $n ./animal.rankings

