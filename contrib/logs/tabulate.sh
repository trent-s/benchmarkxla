#!/usr/bin/bash

# if there are any "time in us" then it is easiest now to edit the original file by hand to make the mean value also in ms...
cat base.txt xla.txt | grep -v _large | grep -v BERT | grep -v cpu |  fgrep 'test_eval[' | tr -d '[],' | grep [0-9] | grep -v '::' | sed -e 's/test_eval//g' | tr -s ' ' | sort | while read a b c d e f g
do
	#echo ${a},${f}
	echo ${a},${f} | sed -e 's/-/,/'
done
# then hand edit resulting file to be consistent for visualization in excel
# removing cpu values helps with comparison... e.g., pipe above to grep -v cpu
