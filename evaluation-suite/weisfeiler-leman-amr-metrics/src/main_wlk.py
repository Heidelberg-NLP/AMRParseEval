import argparse
import numpy as np

def build_arg_parser():

    parser = argparse.ArgumentParser(
            description='Argument parser for WLK')

    parser.add_argument('-a'
            , type=str
            , help='file path to first SemBank')

    parser.add_argument('-b'
            , type=str
            , help='file path to second SemBank')

    parser.add_argument('-log_level'
            , type=int
            , nargs='?'
            , default=40
            , choices=list(range(0, 60, 10))
            , help='logging level (int), see\
                    https://docs.python.org/3/library/logging.html#logging-levels')
    
    parser.add_argument('-k'
            , type=int
            , nargs='?'
            , default=2
            , help='number of WL iterations')
    
    parser.add_argument('-round_decimals'
            , type=int
            , nargs='?'
            , default=3
            , help='decimal places to round scores to. Set to large negative number\
                    to prevent any rounding')

    parser.add_argument('--edge_to_node_transform'
            , action='store_true'
            , help='trasnform to equivalent unlabeled-edge graph, e.g.,\
                    (1, :var, 2) -> (1, :edge, 3), (3, :edge, 2), 3 has label :var')

    parser.add_argument('-output_type'
            , type=str
            , default='score'
            , choices=['score', 'score_corpus', 'score_alignment']
            , help='output options:\
                    score: one score per line for every input graph pair\
                    score_corpus: average score\
                    score_alignment: same as "score" but also provide alignment')

    parser.add_argument('-communication_direction'
            , type=str
            , default='both'
            , choices=['both', 'fromout', 'fromin']
            , help='message passing direction:\
                    both: graph is treated as undirected\
                    fromout: node receive info from -> neighbor ("bottom-up AMR")\
                    fromin: node receive info from <- neighbor ("top-down AMR")')
    
    parser.add_argument('-input_format'
            , type=str
            , nargs='?'
            , default="penman"
            , help='input format: either penman or tsv')
 
    return parser

if __name__ == "__main__":

    import log_helper

    args = build_arg_parser().parse_args()
    logger = log_helper.set_get_logger("WLK simple AMR similarity", args.log_level)
    logger.info("loading amrs from files {} and {}".format(
        args.a, args.b))
    
    import data_helpers as dh
    import amr_similarity as amrsim
    import graph_helpers as gh

    graphfile1 = args.a
    graphfile2 = args.b
    
    grapa = gh.GraphParser(input_format=args.input_format, 
                            edge_to_node_transform=args.edge_to_node_transform)

    string_graphs1 = dh.read_graph_file(graphfile1)
    graphs1, _ = grapa.parse(string_graphs1)

    string_graphs2 = dh.read_graph_file(graphfile2)
    graphs2, _ = grapa.parse(string_graphs2)
   
    predictor = amrsim.WLK(iters=args.k, 
                            communication_direction=args.communication_direction)
 
    def get_scores():
        return predictor.predict(graphs1, graphs2)

    if args.output_type == 'score':
        preds = get_scores()
        preds = np.around(preds, args.round_decimals)
        print("\n".join(str(pr) for pr in preds))

    elif args.output_type == 'score_corpus':
        preds = get_scores()
        print(np.around(np.mean(preds), args.round_decimals))
