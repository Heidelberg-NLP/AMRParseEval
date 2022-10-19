import numpy as np
import scipy.stats as st
from scipy.stats import pearsonr, hmean, gmean, spearmanr, rankdata
from sklearn.metrics import f1_score
import random

random.seed(42)
# SICK and PARA first graph is ignored (empty graph)
LPQUALITY = ("../qualitylp/LABEL.txt", [0, 400])
AMR3QUALITY = ("../qualityamr3/LABEL.txt", [0, 400])

def get_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description='evaluate AMR metric result')
     
    parser.add_argument('-path_princequality_prediction_file_abparser', type=str, nargs='?',
                                help='path to quality prediction file', required = True)
     
    parser.add_argument('-path_amr3quality_prediction_file_abparser', type=str, nargs='?',
                                help='path to quality prediction file', required = True)
    
    return parser


def get_predicted_scores(lines, f = lambda x: x.split()[-1], index=LPQUALITY[0]):
    out = []
    for i,l in enumerate(lines):
        try:
            x = f(l)
            x = float(x)
            out.append(x)
        except:
            out.append("NA")
    return np.array(out[index[0]:index[1]])  


def readl(p):
    with open(p) as f:
        return f.read().split("\n")


def load_parse_quality_lp():
    return load_parse_quality(LPQUALITY)


def load_parse_quality_amr3():
    return load_parse_quality(AMR3QUALITY)


def load_parse_quality(pathnum):
    lines = readl(pathnum[0])[pathnum[1][0]:pathnum[1][1]]
    lines = [l.split("\t") for l in lines[::2]]
    lines = [[float(x) for x in line[:3]] for line in lines]
    pairrank_ab = [l[0] for l in lines] # 1 if a is better 0 if b is better, 0.5 if equal 
    
    def dec(x):
        if x == 0.0:
            return -1
        elif x == 1.0:
            return 1
        elif x == 0.5:
            return 0.0
        raise ValueError("wrong label format")

    pairrank_ab = [dec(l) for l in pairrank_ab] # 1 if a is better -1 if b is better, 0.0 if equal 
    accept_a = [l[0] for l in lines]
    accept_b = [l[1] for l in lines]
    return pairrank_ab, accept_a, accept_b



def rank(xs, start=0):
    arr = rankdata(xs) - start
    return zip(arr, xs)


def evaluate_with_function(xs, ys, fun):
    return fun(xs, ys)

def evaluate_with_classif_function(xs, ys, fun):
    return fun(xs, ys)

def bootstrap(xs, ys, fun):
    results = []
    i = 0
    while i < 10000:
        i += 1
        idxs = list(range(len(xs)))
        tmp = []
        j = 0
        while j < len(xs):
            j += 1
            k = random.choice(idxs)
            tmp.append(k)
        xstmp = [xs[k] for k in tmp]
        ystmp = [ys[k] for k in tmp]
        score = evaluate_with_function(xstmp, ystmp, fun)
        results.append(score)
    return (np.mean(results), np.percentile(results, 5), np.percentile(results, 95))

def safe_evaluate_with_function(xs, ys, fun, do_bootstrap=False):
    ci = None
    try:
        if do_bootstrap:
            mp = evaluate_with_function(xs, ys, fun=fun)
            ci = bootstrap(xs, ys, fun)
        else:
            mp = evaluate_with_function(xs, ys, fun=fun)
    except (ValueError, TypeError) as e:
        print(e)
        mp = "NA"
    return mp, ci



def pair_accuracy(gold, ys):
    correct = 0
    alls = 0
    for i in range(0, len(gold), 2):
        deltag = gold[i] - gold[i + 1]
        deltap = ys[i] - ys[i + 1]
        if deltag * deltap > 0:
            correct += 1
        alls += 1
    return correct / alls



def pw_acc(gold, ys):

    pw = [g * ys[i] for i, g in enumerate(gold)]
    pw = [x for x in pw if x != 0.0]
    
    return len([x for x in pw if x > 0 ]) / len(pw)


def mrank(gold, ys):
    notaccept = [(i, g) for i, g in enumerate(gold) if g == 0.0]
    accept = [(i, g) for i, g in enumerate(gold) if g == 1.0]
    
    rank_score = list(rank(ys))#sorted(enumerate(ys), key=lambda k:k[1])))) 

    
    notaccept_ranks = [rank_score[i][0] for i in [i for i, j in notaccept]]
    accept_ranks = [rank_score[i][0] for i in [i for i, j in accept]]
    
    return np.median(accept_ranks) - np.median(notaccept_ranks)
    




def eval_quality(preda, predb, pairrank_ab, accept_a, accept_b):

    predall = np.concatenate((preda, predb))
    
    accept_score, accept_score_ci = safe_evaluate_with_function(accept_a + accept_b, predall, fun=mrank, do_bootstrap=True)
    pw_a, pw_a_ci = safe_evaluate_with_function(pairrank_ab, np.array(preda) - np.array(predb), fun=pw_acc, do_bootstrap=True) 
    
    ##### for parser ranking we need six values
    ##### 1. how often was a preferred
    ##### 2. how often was b preffered
    ##### 3. delta of 1 and 2
    ##### 4. absolute score for parser 1
    ##### 5. absolute score for parser 2
    ##### 6. delta of 4 and 5

    score1 = len([i for i in np.array(preda) - np.array(predb) if i > 0.0001])
    score2 = len([i for i in np.array(preda) - np.array(predb) if i < 0.0001])
    delta12 = score1 - score2
    
    score3 = np.mean(preda)
    score4 = np.mean(predb)
    delta34 = score3 - score4
    
    parser_scores = [score1, score2, delta12, score3, score4, delta34]
    print(" & ".join([str(round(x, 2)) for x in parser_scores]))


    #print(" & ".join([str(round(x, 2)) for x in [pw_a, accept_score]]))
    
    rs = lambda x : str(round(x, 2)) 
    print("-----------Sys rank------------")
    
    print("parser A # preferences: ", rs(score1))
    print("parser B # preferences: ", rs(score2))
    print("Delta: ", rs(delta12))
    
    print("parser A # macro score: ", rs(score3))
    print("parser B # macro score: ", rs(score4))
    print("Delta: ", rs(delta34))

    print("----------Parse rank-----------")

    print("pairwise accuracy: ", str(round(pw_a, 2)), pw_a_ci)
    print("accept score: ", str(round(accept_score, 2)), accept_score_ci)

    
    return parser_scores, pw_a, accept_score


if __name__ == "__main__":
    
    
    parser = get_arg_parser()
    args = parser.parse_args()
    
    
    #################
    #scores = []
    #weights = []

    ############ Prince quality ############################
    print("Result for Little Prince")
    pairrank_ab, accept_a, accept_b = load_parse_quality_lp()
 
    pathab = args.path_princequality_prediction_file_abparser
    preda = get_predicted_scores(readl(pathab), index=LPQUALITY[1])[::2]
    predb = get_predicted_scores(readl(pathab), index=LPQUALITY[1])[1::2]

    parser_rank, pw_a, accept_score = eval_quality(preda, predb, pairrank_ab, accept_a, accept_b)

    print("\n\n---------------------------\n\n")
    ############ amr3 quality ############################
    print("Result for AMR3")
    pairrank_ab, accept_a, accept_b = load_parse_quality_amr3()
 
    pathab = args.path_amr3quality_prediction_file_abparser
    preda = get_predicted_scores(readl(pathab), index=AMR3QUALITY[1])[::2]  
    predb = get_predicted_scores(readl(pathab), index=AMR3QUALITY[1])[1::2]
    
    parser_rank, pw_a, accept_score = eval_quality(preda, predb, pairrank_ab, accept_a, accept_b)
    

    #print("Latex --->",  "\multicolumn{9}{c}{Pearson's $\\rho$} & \multicolumn{9}{c}{Accuracy} & amean & gmean & hmean & SAMPLE WEIGHTED MEAN" )
    #print("Latex --->",  eval_rep(scores, weights=np.array(weights)))
    #print(" | --->",  eval_rep(scores, weights=np.array(weights), combiner=" | "))



