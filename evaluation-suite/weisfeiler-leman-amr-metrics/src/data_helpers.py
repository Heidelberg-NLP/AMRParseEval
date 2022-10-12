def clean_from_meta(amrstring):
    strings = amrstring.split("\n")
    return "\n".join([l for l in strings if not l.startswith("#")])

def read_graph_file(p):
    """reads linearized string amr graphs from amr sembank.
    
    Args:
        p (list): path to AMR sembank that contains AMR graphs, i.e.
                                # ::snt ...
                                (x / ...
                                    :a (...))

                                # ::snt ...
                                (...

    Returns:
        - list with string AMRs in Penman format
    """

    with open(p) as f:
        amrs = [l for l in f.read().split("\n\n") if l]
    amrs = [clean_from_meta(string) for string in amrs]
    amrs = [amr for amr in amrs if amr]
    return amrs

def read_score_file(p):
    with open(p) as f:
        scores = [float(s) for s in f.read().split("\n")]
    return scores

def write_string_to_file(string, p):
    with open(p, "w") as f:
        f.write(string)

