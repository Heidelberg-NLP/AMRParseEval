# SEMA
This repository contains source code of SEMA, a metric to evaluate AMR
```
@inproceedings{anchieta2019sema,
    title={SEMA: an Extended Semantic Evaluation for AMR},
    author={Anchi\^{e}ta, Rafael Torres and Cabezudo, Marco Antonio Sobrevilla and Pardo, Thiago Alexandre Salgueiro},
    booktitle={(To appear) Proceedings of the 20th Computational Linguistics and Intelligent Text Processing},
    year={2019},
    editor={Gelbukh, Alexander},
    publisher={Springer International Publishg}
}
```


# Requirements
Python (version 3.6 or later)

# Usage
`python sema.py -t parsed_file.txt -g reference_file.txt`

# Input files format
Test and Reference AMR files should be in PENMAN format.
```
(t / tolerate-01
    :ARG0 (w / we)
    :ARG1 (c / coutry
        :name (n / name :op1 "Japan")
        :wiki "Japan")
    :duration (a / amr-unknown))
```
