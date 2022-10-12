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
    python smatch.py -f $CURR/../$dat/$task/tgt.test.amr $CURR/../$dat/$task/src.test.amr --ms > $CURR/sim-predictions/$dat-smatch-$task.txt
    done
done

cd $CURR
