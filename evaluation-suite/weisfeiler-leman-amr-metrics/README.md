# Weisfeiler-Leman Graph Kernels for AMR Graph Similarity

The repository contains python code for metrics of AMR graph similarity.

**New in Version 0.2**: faster, more options, increase stabiltiy, other graph formats.

## Requirements

Install the following python packages (with pip or conda):

```
numpy (tested: 1.19.4)
scipy (tested: 1.1.0) 
networkx (tested: 2.5)
gensim (tested: 3.8.3)
penman (tested: 1.1.0)
pyemd (tested: 0.5.1)
```

## Computing AMR metrics

### Basic Wasserstein AMR similarity

```
cd src
python main_wlk_wasser.py -a <amr_file> -b <amr_file>
```

Note that the node labels will get initialized with GloVe vectors, 
it can take a minute to load them. If everything should be randomly intitialzed 
(no loading time), set `-w2v_uri none`.

### Return AMR n:m alignment:

```
cd src
python main_wlk_wasser.py -a <amr_file> -b <amr_file> -output_type score_alignment
```

This prints the scores and many-many node alignments, with flow and cost information. 

### Learning edge weights

```
cd src
python main_wlk_wasser_optimized.py -a_train <amr_file> -b_train <amr_file> \
                                    -a_dev <amr_file> -b_dev <amr_file> \
                                    -a_test <amr_file> -b_test <amr_file> \
                                    -y_train <target_file> -y_dev <target_file>
```

where `<target_file>` is a file the contains a float per line for which we
optimize the parameters. In the end the script will return predictions for
`-a_test <amr_file>` vs. `b_test <amr_file>`.


### Symbolic (Structural) AMR similarity

```
cd src
python main_wlk.py -a <amr_file> -b <amr_file>
```

## Notes

### Increase numerical stability

Currently, only `main_wlk.py`, i.e., the structural WLK provides fully deterministic results.
Since in current Wasserstein WLK the edges and words not in GloVe are initialized randomly, 
it can lead to some variation in the predictions. More stable results for WWLK and alignments are desired, 
consider using the new `-stability_level` parameter, e.g.:

```
cd src
python main_wlk_wasser.py -a <amr_predicted> -b <amr_ref> \
                          -stability_level 15
```

It computes an expected contextualized node distance matrix by repeated sampling of any unknown random parameters (`-stability_level n` samples), before calculating the Wasserstein distance.

### Parsing evaluation

Use `-stability_level` for increased stability when using wasser wlk (as above). And calculate a corpus score (currently, only output option is the mean over all scores). A good option for parsing evaluation may be:

```
python -u main_wlk_wasser.py -a <amr_predicted> -b <amr_ref> \
                              -output_type score_corpus \
                              -stability_level 15 -k 4 \
                              -random_init_relation constant \
                              --edge_to_node_transform 
```

This also transforms the graphs to (equivalent) graphs with unlabeled edges (see below), and lets us set constant edge weights.

### Processing graphs other than AMR

You can use the metrics for comparing/aligning other graph-based meaning representations, and node-labeled graphs in general.

For graphs other than AMR use `-input_format tsv`. Then you can input files with tab or whitespace speparated triples. An empty line indicates begin of another graph. The general format is `<src_node_id> <tgt_node_id> <relation_label>`, node labels are indicated with `:instance` triples. A graph looks similar to:

```
n1 n2 :rel_a
n2 n3 :rel_b
n1 label_n1 :instance
n2 label_n2 :instance
n3 label_n3 :instance
```

This graph contains 3 nodes and 2 edges, all edges and nodes have labels. That's it!

### Score range

For convenience, score range is now in [-1, 1] for minimum similarity (-1) and maximum similarity (1).

### Important Options

Some important options that can be set according to use-case

- `-w2v_uri <string>`: use different word embeddings (FastText, word2vec, etc.). Current default: `glove-wiki-gigaword-100`.
- `-k <int>`: Use an int to specify the maximum contextualization level. E.g., If k=5, a node will receive info from nbs that are up to 5 hops away.
- `-stability_level <int>`: Consider two graphs with a few random parameters. We calculate the expected node distance matrix by sampling parameters `<int>` times. This increases stability of results but also increases runtime. A good trade-off may be 10 or 20. 
- `-communication_direction <string>`: There are three options. Consider (x, :arg0, y), where :arg0 is directed. Option `fromin` means y receives from x. Option `fromout` means that x receives from y. `both` (default value) means `fromin` *and* `fromout`.
- `--edge_to_node_transform`: This flag transforms the edge-labeled AMR graph into an (equivalent) graph with unlabeled edges. E.g., (1, arg1, 2), (1, arg2, 3) --> (1, 4), (1, 5), (4, 2), (5, 3), where 4 has label arg1 and 5 has label arg2.


More options can be checked out:

```
cd src
python main_wlk_wasser.py --help
```

### Benchmarking

Some scores on BAMBOO of current configurations: see `info/`

Approx Processing TIME 1000 graph pairs: 

| method | time (seconds) |
|  ----  |  ------------- |
| WLK    | 0.5            |
| WWLK   | 5              |

1000 graph pairs WWLK need 5 seconds

## Version notes

- 0.1: initial release
- 0.2: speed increase by making better use of numpy, more stability by distance matrix sampling, labeled to unlabeld graph transform, refactored code

## Citation

```
@article{10.1162/tacl_a_00435,
    author = {Opitz, Juri and Daza, Angel and Frank, Anette},
    title = "{Weisfeiler-Leman in the Bamboo: Novel AMR Graph Metrics and a Benchmark for AMR Graph Similarity}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {9},
    pages = {1425-1441},
    year = {2021},
    month = {12},
    abstract = "{Several metrics have been proposed for assessing the similarity of (abstract) meaning representations (AMRs), but little is known about how they relate to human similarity ratings. Moreover, the current metrics have complementary strengths and weaknesses: Some emphasize speed, while others make the alignment of graph structures explicit, at the price of a costly alignment step.In this work we propose new Weisfeiler-Leman AMR similarity metrics that unify the strengths of previous metrics, while mitigating their weaknesses. Specifically, our new metrics are able to match contextualized substructures and induce n:m alignments between their nodes. Furthermore, we introduce a Benchmark for AMR Metrics based on Overt Objectives (Bamboo), the first benchmark to support empirical assessment of graph-based MR similarity metrics. Bamboo maximizes the interpretability of results by defining multiple overt objectives that range from sentence similarity objectives to stress tests that probe a metric’s robustness against meaning-altering and meaning- preserving graph transformations. We show the benefits of Bamboo by profiling previous metrics and our own metrics. Results indicate that our novel metrics may serve as a strong baseline for future work.}",
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00435},
    url = {https://doi.org/10.1162/tacl\_a\_00435},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00435/1979290/tacl\_a\_00435.pdf},
}

``` 
