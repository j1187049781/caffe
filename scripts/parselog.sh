#!/bin/bash
# Usage parselog.sh caffe.log 
# It creates two files one caffe.log.test that contains the loss and test accuracy of the test and
# another one caffe.log.loss that contains the loss computed during the training

if [ "$#" -lt 1 ]
then
echo "Usage parselog.sh /path_to/caffe.log"
fi
LOG=`basename $1`
grep -B 2 'Test ' $1 > aux.txt
grep 'Iteration ' aux.txt | sed  's/.*Iteration \([[:digit:]]*\).*/\1/g' > aux0.txt
grep 'Test score #0' aux.txt | awk '{print $8}' > aux1.txt
grep 'Test score #1' aux.txt | awk '{print $8}' > aux2.txt
grep ' loss =' $1 | awk '{print $6,$9}' | sed 's/,//g' | column -t > $LOG.loss 
paste aux0.txt aux1.txt aux2.txt | column -t > $LOG.test
rm aux.txt aux0.txt aux1.txt aux2.txt