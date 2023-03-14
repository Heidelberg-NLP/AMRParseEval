import sys
import re

def read_amr_file(p1):
    with open(p1, "r") as f:
        amrs = f.read().split("\n\n")
    sents = [amr.split("# ::snt")[1].split("\n")[0] for amr in amrs]
    amrs = [amr.split("#")[-1].split("\n", 1)[1] for amr in amrs]
    amrs = [amr.replace("\n", " ") for amr in amrs]
    amrs = [" ".join(amr.split()) for amr in amrs]
    return amrs, sents


def sent_lens(sents):
    
    return [len(s.split()) for s in sents]

def simple_overlap(amrs1, amrs2):
    
    def get_nodes_edges(amr):
        amr = re.sub(r"\([a-z][a-z]?[0-9]*", " ", amr)
        amr = re.sub(r" [a-z][a-z]?[0-9]* ", " ", amr)
        amr = amr.replace("(", " ")
        amr = amr.replace(")", " ")
        amr = amr.replace("/", " ")
        toks = amr.replace("\n", " ").split()
        return toks

    def so(a1, a2):
        toks_1 = get_nodes_edges(a1)
        toks_2 = get_nodes_edges(a2)
        intersec = set(toks_1).intersection(set(toks_2))
        union = set(toks_1).union(set(toks_2))
        score =  len(intersec) / len(union)
        return score
    
    out = []
    for i, a1 in enumerate(amrs1):
        out.append(so(a1, amrs2[i]))
    
    return out


a1, s = read_amr_file(sys.argv[1])
a2, _ = read_amr_file(sys.argv[2])

result = simple_overlap(a1, a2)

print("\n".join([str(num) for num in result]))

