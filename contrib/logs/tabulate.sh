#!/usr/bin/bash

# if there are any "time in us" then it is easiest now to edit the original file by hand...
cat base.txt xla.txt | fgrep 'test_eval[' | tr -d '[],' | grep [0-9] | grep -v '::' | sed -e 's/test_eval//g' | tr -s ' ' | sort | while read a b c d e f g
do
	echo ${a},${f}
done
# then hand edit resulting file to be consistent for visualization
