CURR=$(pwd)
echo $CURR

SRC=amr-metric-suite/py3-Smatch-and-S2match/smatch/

cd $SRC


for dat in qualitylp qualityamr3
do
    for task in abparser
    do
    echo $task
    echo $dat
    python s2match.py -f $CURR/../$dat/$task/tgt.test.amr $CURR/../$dat/$task/src.test.amr -vectors ../../vectors/glove.6B.100d.txt --ms > $CURR/sim-predictions/$dat-s2match-$task.txt
    done
done

cd $CURR
