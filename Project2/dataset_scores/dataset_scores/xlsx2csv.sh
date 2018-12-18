#!/bin/bash

for f in `ls *.xlsx`; do
  NAME=`echo "$f" | cut -d'.' -f1`
  in2csv ${NAME}.xlsx > ${NAME}.csv
done

