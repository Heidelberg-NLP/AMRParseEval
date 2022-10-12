CURR=$(pwd)
echo $CURR

SRC=weisfeiler-leman-amr-metrics/src/

cd $SRC


for dat in qualitybio qualitylp qualityamr3
do
    for task in abparser
    do
    echo $task
    echo $dat
	python main_wlk_wasser.py -a $CURR/../$dat/$task/tgt.test.amr -b $CURR/../$dat/$task/src.test.amr -stability_level 15 -k 2 -round_decimals 10 > $CURR/sim-predictions/$dat-wwlk2-$task.txt    
done
done

cd $CURR
