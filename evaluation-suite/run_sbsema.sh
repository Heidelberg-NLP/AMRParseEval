CURR=$(pwd)
echo $CURR



SB=sembleu/src
SE=sema


cd $SB


for k in {1..4}
do
for dat in qualitylp qualityamr3
do
    for task in abparser
    do
    echo $task
    echo $dat
    python eval2.py $CURR/../$dat/$task/src.test.amr $CURR/../$dat/$task/tgt.test.amr $k > $CURR/sim-predictions/$dat-sembleuST-k$k-$task.txt
    done
done
done

cd $CURR

cd $SE


for dat in qualitylp qualityamr3
do
    for task in abparser
    do
    echo $task
    echo $dat
    python sema.py -t $CURR/../$dat/$task/tgt.test.amr -g $CURR/../$dat/$task/src.test.amr > $CURR/sim-predictions/$dat-sema-$task.txt    
done
done


cd $CURR
