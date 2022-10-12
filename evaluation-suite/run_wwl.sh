CURR=$(pwd)
echo $CURR

SRC=weisfeiler-leman-amr-metrics/src/

cd $SRC

#for dat in sts sick para
#do
#    for task in reify main role_confusion syno
#    do
#    echo $task
#    echo $dat
#    python main_wlk_wasser.py -a $CURR/../$dat/$task/tgt.test.amr -b $CURR/../$dat/$task/src.test.amr -stability_level 15 -k 4 --edge_to_node_transform -round_decimals 10 -random_init_relation constant > $CURR/sim-predictions/$dat-wwlk4e2n-$task.txt
#    done
#done

for dat in qualitybio qualitylp qualityamr3
do
    for task in abparser
    do
    echo $task
    echo $dat
	python main_wlk_wasser.py -a $CURR/../$dat/$task/tgt.test.amr -b $CURR/../$dat/$task/src.test.amr -stability_level 15 -k 3 --edge_to_node_transform -round_decimals 10 -random_init_relation ones > $CURR/sim-predictions/$dat-wwlk3e2n-$task.txt    
done
done

cd $CURR
