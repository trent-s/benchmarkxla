#!/usr/bin/bash

#  cat table1.csv | ./flatten.sh >table2.csv


IFS=,
echo "model,cuda-eager,cuda-jit,xla-eager,xla-jit"
while true
	do
	read a1 b1 c1
	if [[ ${a1} == "" ]]
	then
		break
	fi
	read a2 b2 c2
	read a3 b3 c3
	read a4 b4 c4
	echo ${a1},${c1},${c2},${c3},${c4}
	done
