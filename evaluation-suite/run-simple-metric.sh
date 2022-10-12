CURR=$(pwd)
echo $CURR

name=simpleMetric
#name=sentLen


for dat in qualitylp qualityamr3
do
    for task in abparser 
    do
    python simple_metric.py ../$dat/$task/tgt.test.amr ../$dat/$task/src.test.amr > sim-predictions/$dat-$name-$task.txt 
    done
done
