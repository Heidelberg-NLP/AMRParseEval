import argparse

def build_arg_parser():

    parser = argparse.ArgumentParser(
            description='Argument parser for optimizing WWLK edge weights')

    parser.add_argument('-a_train'
            , type=str
            , help='file path to first train SemBank')

    parser.add_argument('-b_train'
            , type=str
            , help='file path to second train SemBank')
     
    parser.add_argument('-a_dev'
            , type=str
            , help='file path to first dev SemBank')

    parser.add_argument('-b_dev'
            , type=str
            , help='file path to second dev SemBank')
    
    parser.add_argument('-y_dev'
            , type=str
            , nargs="?"
            , default=None
            , help='file path to second dev target score')
    
    parser.add_argument('-y_train'
            , type=str
            , nargs="?"
            , default=None
            , help='file path to train target score')
     
    parser.add_argument('-a_test'
            , type=str
            , help='file path to first test SemBank')

    parser.add_argument('-b_test'
            , type=str
            , help='file path to second test SemBank')

    parser.add_argument('-log_level'
            , type=int
            , nargs='?'
            , default=20
            , choices=list(range(0, 60, 10))
            , help='logging level (int), see\
                    https://docs.python.org/3/library/logging.html#logging-levels')
    
    parser.add_argument('-w2v_uri'
            , type=str
            , nargs="?"
            , default="glove-wiki-gigaword-100"
            , help='string with w2v uri, see gensim docu, e.g.\
                    \'word2vec-google-news-300\'')

    parser.add_argument('-k'
            , type=int
            , nargs='?'
            , default=2
            , help='number of WL iterations') 
    
    parser.add_argument('-init_lr'
            , type=float
            , nargs='?'
            , default=0.75
            , help='initial learning rate') 

    parser.add_argument('--edge_to_node_transform'
            , action='store_true'
            , help='trasnform to equivalent unlabeled-edge graph, e.g.,\
                    (1, :var, 2) -> (1, :edge, 3), (3, :edge, 2), 3 has label :var')
    
    parser.add_argument('-input_format'
            , type=str
            , nargs='?'
            , default="penman"
            , help='input format: either penman or tsv') 
    
    return parser

if __name__ == "__main__":
        
    import log_helper
    
    args = build_arg_parser().parse_args()
    logger = log_helper.set_get_logger("Wasserstein AMR similarity", args.log_level)
    logger.info("loading amrs from files {} and {}".format(
        args.a_train, args.b_train))
    
    import black_box_optim as optim
    import data_helpers as dh
    import amr_similarity as amrsim
    import graph_helpers as gh

    graphfile1_train = args.a_train
    graphfile2_train = args.b_train
    
    
    graphfile1_dev = args.a_dev
    graphfile2_dev = args.b_dev
    
    graphfile1_test = args.a_test
    graphfile2_test = args.b_test
    
    grapa = gh.GraphParser(input_format=args.input_format, 
                            edge_to_node_transform=args.edge_to_node_transform) 
    #################

    string_graphs1_train = dh.read_graph_file(graphfile1_train)
    graphs1_train, node_map1_train = grapa.parse(string_graphs1_train)

    string_graphs2_train = dh.read_graph_file(graphfile2_train)
    graphs2_train, node_map2_train = grapa.parse(string_graphs2_train)
    
    prepro = amrsim.AmrWasserPreProcessor(w2v_uri=args.w2v_uri, is_resettable=False)
    prepro.prepare(graphs1_train, graphs2_train)
    
    ################

    string_graphs1_dev = dh.read_graph_file(graphfile1_dev)
    graphs1_dev, node_map1_dev = grapa.parse(string_graphs1_dev)

    string_graphs2_dev = dh.read_graph_file(graphfile2_dev)
    graphs2_dev, node_map2_dev = grapa.parse(string_graphs2_dev)
    
    
    ################

    string_graphs1_test = dh.read_graph_file(graphfile1_test)
    graphs1_test, node_map1_test = grapa.parse(string_graphs1_test)

    string_graphs2_test = dh.read_graph_file(graphfile2_test)
    graphs2_test, node_map2_test = grapa.parse(string_graphs2_test)
    
    
    ################

    predictor = amrsim.WasserWLK(preprocessor=prepro, iters=args.k) 
    predictor.predict(graphs1_train[:2], graphs2_train[:2]) 
    
    # if training and dev targets not exists then it is role confusion 
    # which means targets are 0, 1, 0, 1, 0, 1 ....
    if not args.y_train:
        targets = [0.0, 1.0] * len(string_amrs1_train)
        targets = targets[:len(string_amrs1_train)]
    else:
        targets = dh.read_score_file(args.y_train) 
   
    if not args.y_dev:
        targets_dev = [0.0, 1.0] * len(string_amrs1_dev)
        targets_dev = targets_dev[:len(string_amrs1_dev)]
    else:
        targets_dev = dh.read_score_file(args.y_dev) 
    
    
    optimizer = optim.SPSA(graphs1_train, graphs2_train, predictor
                            , targets, dev_graphs_a=graphs1_dev
                            , dev_graphs_b=graphs2_dev, targets_dev=targets_dev
                            , init_lr=args.init_lr, eval_steps=100)
    
    optimizer.fit()
    preds = predictor.predict(graphs1_test, graphs2_test)

    print("\n".join(str(pr) for pr in preds))
    
