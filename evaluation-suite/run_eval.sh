metric=$1

python -u evaluate4tasks.py -path_princequality_prediction_file_abparser sim-predictions/qualitylp-$metric-abparser.txt \
    -path_amr3quality_prediction_file_abparser sim-predictions/qualityamr3-$metric-abparser.txt 
