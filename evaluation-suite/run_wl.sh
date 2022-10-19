CURR=$(pwd)
echo $CURR

SRC=weisfeiler-leman-amr-metrics/src/

cd $SRC


for dat in qualitylp qualityamr3
do
    for task in abparser
    do
    echo $task
    echo $dat
    python main_wlk.py -a $CURR/../$dat/$task/tgt.test.amr -b $CURR/../$dat/$task/src.test.amr > $CURR/sim-predictions/$dat-wlk2-$task.txt
    done
done

cd $CURR
