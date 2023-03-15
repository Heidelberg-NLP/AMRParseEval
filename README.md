# AMR parse(r) evaluation experiment

This is code/data for our paper **Better Smatch = better parser? AMR evaluation is not so simple anymore**

## Data

Data for little prince:

- `qualitylp/abparser/src.test.amr`: pairs of candidate parses (first parse is of BART based parser, second parse by T5 based parser)
- `qualitylp/abparser/tgt.test.amr`: pairs of reference parses (first and second parse are the same)
- `qualitylp/LABEL.txt`: contains human annotation for the pairs. First number: preference (1: prefer first, 0: prefer second, 0.5: both are equal quality). Second number: is first parse acceptable? 1: Yes, 0: No. Third number: Is second parse acceptable? 1: yes, 0: No

## Data for AMR3:

Is found in `qualityamr3`. It follows the same format as little prince data. **Note**: Release of reference graphs requires LDC license, therefore this repo does not contain `qualityamr3/abparser/tgt.test.amr`. For constructing the reference AMR3, please see also *Notes* below.

## Run experiments

Please look in this directory:

`evaluation-suite`

## Notes

- Removal of `:wiki` relations: Since parsers usually use external post-precessing for wiki linking, we do not consider wiki relations for AMR quality. We added a script for wiki removal in `util/format_amr.py`. When constructing the reference for AMR3, it has to be applied.

- Re-naming variables: We standardized variables with a running index and prevent occurence of a variable that is also a concept. See also the script in `util/format_amr.py`. This is internally also done by some, but not all metrics.

## Citation

If you find the work interesting, consider citing

```
@inproceedings{opitz-frank-2022-better,
    title = "Better {S}match = Better Parser? {AMR} evaluation is not so simple anymore",
    author = "Opitz, Juri and Frank, Anette",
    booktitle = "Proceedings of the 3rd Workshop on Evaluation and Comparison of NLP Systems",
    month = nov,
    year = "2022",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.eval4nlp-1.4",
    doi = "10.18653/v1/2022.eval4nlp-1.4",
    pages = "32--43",
}
```


