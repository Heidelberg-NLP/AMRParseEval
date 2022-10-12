Some results on [BAMBOO](https://github.com/flipz357/bamboo-amr-benchmark). Resuls for STS, SICK, PARA. And an arithmetic mean and sample weighted arithmetic mean that take results on robustness challenges into account. Sample weighted mean assigns more weight to scores from data sets where more data is available (depending on data size).

| Metric      | STS   | SICK  | PARA  | STS(reify) | SICK(reify) | PARA(reify)| STS(Syno) | SICK(Syno) | PARA(Syno) | STS(role) | SICK(role) | PARA(role) | AMEAN | GMEAN | HMEAN |WMEAN |
|-------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Smatch      | 58.39 | 59.75 | 41.32 | 58.03 | 61.79 | 39.47 | 56.13 | 57.37 | 39.54 | 89.87 | 98.32 | 88.14 | 62.34 | 59.64 | 57.14 | 58.44 |
| S2match(def)| 56.39 | 58.11 | 42.40 | 55.78 | 59.97 | 40.67 | 56.04 | 57.15 | 40.93 | 93.67 | 98.32 | 91.26 | 62.56 | 59.78 | 57.33 | 58.06 |
| S2match     | 58.70 | 60.47 | 42.52 | 58.19 | 62.37 | 40.55 | 56.62 | 57.88 | 41.15 | 89.87 | 98.32 | 92.24 | 63.24 | 60.58 | 58.15 | 59.42 |
| Sema        | 55.90 | 53.32 | 33.43 | 55.51 | 56.16 | 32.33 | 50.16 | 48.87 | 29.11 | 78.48 | 90.76 | 74.93 | 54.91 | 51.91 | 48.95 | 51.20 |
| SemBleu(k3) | 56.54 | 58.06 | 32.82 | 54.96 | 58.53 | 33.66 | 53.19 | 53.72 | 28.96 | 81.01 | 93.28 | 77.79 | 56.88 | 53.64 | 50.37 | 53.88 |
| WLK-k2      | 65.26 | 61.37 | 36.13 | 63.45 | 62.53 | 36.20 | 59.87 | 56.41 | 32.37 | 78.48 | 90.76 | 78.41 | 60.10 | 57.36 | 54.42 | 57.57 |
| WWLK-k2     | 67.35 | 67.56 | 38.16 | 64.41 | 67.16 | 37.08 | 62.14 | 61.88 | 34.02 | 87.34 | 97.48 | 67.62 | 62.68 | 59.81 | 56.77 | 60.35 |
| WWLK-k4 e2n | 64.40 | 66.88 | 37.56 | 63.74 | 66.89 | 37.33 | 60.42 | 61.82 | 33.90 | 91.14 | 100.00 | 93.22 | 64.78 | 61.22 | 57.57 | 62.05 |

Note:
- X(role) are measured with accuracy, everything else in Pearson rho, global aggregate metrics via (P)MEAN
- Smatch: standard Smatch.
- S2match def.: S2match with default HPs `-cutoff: 0.5`, `-diffsense: 0.5`. GloVe Embeddings.
- S2match: S2match with HPs `-cutoff: 0.9`, `-diffsense: 0.95`. GloVe Embeddings.
- Sema: from their repo.
- SemBleu: from their repo.
- WLK-k2: Structural Weisfeiler Leman AMR kernel. Contextualize nodes over iterations, collect graph signatures from iterations, cosine similarity. Call with `python main_wlk.py -a <path1> -b <path2>`
- WWLK-k2: Wasserstein Weisfeiler Leman AMR kernel. Contextualize nodes in latent space over iterations, collect node embeddings, Wasserstein distance. Call with `python main_wlk_wasser.py -a <path1> -b <path2> -stability_level 15`.
- WWLK-k4-e2n: More stable version for more consistent parsing evaluation results. Transforms edge-labeled AMR graph to equivalent graph without edge labels, where edges are now nodes. Call with: `python main_wlk_wasser.py -a <path1> -b <path2> -stability_level 15 -k 4 --edge_to_node_transform -round_decimals 10 -random_init_relation constant`

