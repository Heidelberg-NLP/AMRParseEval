# AMR parse(r) evaluation experiment

This is code/data for our paper **Better Smatch = better parser? AMR evaluation is not so simple anymore**

# Data

Data for little prince:

- `qualitylp/abparser/src.test.amr`: pairs of candidate parses (first parse is of BART based parser, second parse by T5 based parser)
- `qualitylp/abparser/tgt.test.amr`: pairs of reference parses (first and second parse are the same)
- `qualitylp/LABEL.txt`: contains human annotation for the pairs. First number: preference (1: prefer first, 0: prefer second, 0.5: both are equal quality). Second number: is first parse acceptable? 1: Yes, 0: No. Third number: Is second parse acceptable? 1: yes, 0: No

Data for AMR3:

Is found in `qualityamr3`. It follows the same format as little prince data. **Note**: Release of reference graphs requires LDC license, therefore this repo does not contain `qualitylp/abparser/tgt.test.amr`.

# Run experiments

Please look in this directory:

`evaluation-suite`

# Citation

To appear



