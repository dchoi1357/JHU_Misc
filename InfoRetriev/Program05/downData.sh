#!/usr/bin/env bash

if [ "$#" -ne 1 ] || ! [ -d "$1" ]; then
	echo "Usage: $0 [destDir]"
	exit 2
fi

curl "http://pmcnamee.net/744/data/19991220-Excite-QueryLog.utf8.tsv.gz" > \
	"$1/19991220-Excite-QueryLog.utf8.tsv.gz"
gunzip "19991220-Excite-QueryLog.utf8.tsv.gz"
