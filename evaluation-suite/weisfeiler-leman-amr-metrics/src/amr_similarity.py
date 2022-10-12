import logging
import multiprocessing
import re
import numpy as np
from scipy.spatial.distance import cosine, cdist
from scipy.stats import entropy
from pyemd import emd_with_flow
import gensim.downloader as api
import networkx as nx

np.random.seed(42)

logger = logging.getLogger(__name__)


class GraphSimilarityPredictor():
    """(Interface): predicts similarities for paired Inputs that are multi edge networkx Di-Graphs,
    """

    def validate(self, graphs):
        vals = [isinstance(g, nx.MultiDiGraph) for g in graphs]
        if not all(vals):
            raise ValueError("wrong input type, need networkx multi-edge directed graphs")
        return None

    def predict(self, graphs_1, graphs_2):
        return self._predict(graphs_1, graphs_2)


class GraphSimilarityPredictorAligner(GraphSimilarityPredictor):
    """(Interface): predicts similarities for paired amrs Input are multi edge networkx Di-Graphs 
    """
    
    def validate(self, graphs, nodemaps):
        vals = [isinstance(g, nx.MultiDiGraph) for g in graphs]
        if not all(vals):
            raise ValueError("wrong input type, need networkx multi-edge directed graphs")
        vals = [isinstance(nm, dict) for nm in nodemaps]
        if not all(vals):
            raise ValueError("wrong input type, need dictionaries")
        return None

    def predict_and_align(self, graphs_1, graphs_2, node_map1, node_map2):
        return self._predict_and_align(graphs_1, graphs_2, node_map1, node_map2)


class Preprocessor():
    """(Interface): preprocesses data of paired multi edge networkx Di-Graphs 
    """

    def transform(self, graphs_1, graphs_2):
        """preprocesses amr graphs"""
        self._transform(graphs_1, graphs_2)
        return None
    
    def prepare(self, graphs_1, graphs_2):
        self._prepare(graphs_1, graphs_2)
        return None
    
    def reset(self):
        self._reset()
        return None

 
class AmrWasserPreProcessor(Preprocessor):

    def __init__(self, w2v_uri='glove-wiki-gigaword-100', 
                    relation_type="scalar", init="random_uniform",
                    is_resettable=True):
        """Initilize Preprocessor object

        Args:
            w2v_uri (string): uri to desired word embedding
                              e.g., 
                              'word2vec-google-news-100'
                              'glove-twitter-100'
                              'fasttext-wiki-news-subwords-300'
                              etc.
                              if None, then use only random embeddings
                              
                              alternatively: a dict type object with 
                              pretrained vecs
            relation_type (string): edge label representation type
                                    possible: either 'scalar' or 'vector'
            init (string): how to initialize edge weights?
            is_resettable (bool): can the parameters be resetted?
        """

        w2v_uri = w2v_uri
        self.dim = 100
        self.wordvecs = {}
        self.relation_type = relation_type
        self.params_ready = False
        self.init = init
        self.param_keys = None
        self.params = None
        self.unk_nodes = {}
        self.is_resettable = is_resettable
        if w2v_uri:
            if not isinstance(w2v_uri, str):
                self.wordvecs = w2v_uri
                self.dim = self.wordvecs["dog"].shape[0] 
            try:
                self.wordvecs = api.load(w2v_uri)
                self.dim = self.wordvecs["dog"].shape[0]
            except ValueError:
                logger.warning("tried to load word embeddings specified as '{}'.\
                        these are not available. all embeddings will be random.\
                        If this is desired, no need to worry.".format(w2v_uri))
        return None

    def _prepare(self, graphs_1, graphs_2):
        """initialize embedding of graph nodes and edges"""
        self.params_ready = False
        self._xprepare(graphs_1, graphs_2)
        self.params_ready = True
        return None

    def _reset(self):
        self.params_ready = False
        self.param_keys = None
        self.params = None
        self.unk_nodes = {}
        return None

    def _transform(self, graphs1, graphs2):
        """embeds nx multi-digraphs. I.e. assigns every 
        node and edge attribute "latent" with a parameter

        Args:
            graphs1 (list): list withnx medi graph
            graphs2 (list): list with nx medi graph 

        Returns:
            nx multi-edge di graphs with embedded nodes, node map from
            graph nodes to original AMR variable nodes
        """
          
        #gather embeddings for nodes and edges
        self.embeds(graphs1, graphs2)

        return None
     
    def embeds(self, gs1, gs2):
        """ embeds all graphs, i.e., assign embeddings to node labels 
            and edge labels

        Args:
            gs1 (list with nx medi graphs): a list of graphs
            gs2 (list with nx medi graphs): a list of graphs

        Returns:
            None
        """

        for g in gs1:
            self.embed(g)
        for g in gs2:
            self.embed(g)
        logger.debug("no embeddings available for node labels: {}".format(self.unk_nodes.keys()))
        return None
    
    def embed(self, G):

        #get unknown nodes 
        label_no_vec = set()
        label_vec = {}
        for node in G:
            label = G.nodes[node]["label"]
            if label in self.unk_nodes:
                label_vec[label] = self.unk_nodes[label]
                continue
            vec = self._get_vec(label)
            if vec is None:
                label_no_vec.add(label)
            else:
                label_vec[label] = vec
        
        rand_vecs = np.random.uniform(-0.05, 0.05, size=(len(label_no_vec), self.dim))
        
        for i, label in enumerate(label_no_vec):
            vec = rand_vecs[i]
            self.unk_nodes[label] = vec
            label_vec[label] = vec
        
        #set node latent
        def _make_latent(graph):
            for node in graph.nodes:
                label = graph.nodes[node]["label"]
                graph.nodes[node]["latent"] = label_vec[label]
            return None

        _make_latent(G)

        return None
      
    def _get_vec(self, string):
        """lookup a vector for a string
        
        Args:
            string (str): a string
        
        Returns: 
            n-dimensional numpy vector
        """ 
        string = string.strip()

        # if the node is a negation node in AMR (indicated as '-', we assign
        # random vec)
        if string == "-":
            string = "false-not-untrue"

        # further cleaning
        string = string.replace("\"", "").replace("'", "")
        string = string.lower()

        # we can delete word senses here (since they will get contextualized)
        string = re.sub(r'-[0-9]+', '', string)
        
        string = string.replace(":", "")
        
        vecs = [] 
         
        #lookup
        for key in string.split("-"):
            if key in self.wordvecs:
                vecs.append(np.copy(self.wordvecs[key]))
        if "_" in string:
            for key in string.split("_"):
                if key in self.wordvecs:
                    vecs.append(np.copy(self.wordvecs[key]))
        if vecs:
            return np.mean(vecs, axis=0)
        
        
        if string.isnumeric():
            chars = list(string)
            for i, key in enumerate(chars):
                if key in self.wordvecs:
                    vecs.append(np.copy(self.wordvecs[key]) / (1+i))
        if vecs:
            return np.mean(vecs, axis=0)

        return None

    def _xprepare(self, graphs1, graphs2):
        """Prepares the edge label parameters.

        Args:
            graphs1 (list with nx medi graphs): a list with graphs
            graphs2 (list with nx medi graphs): a list with graphs

        Returns:
            None
        """

        es = [self.get_edge_labels(g) for g in graphs1]
        es += [self.get_edge_labels(g) for g in graphs2]
        edge_labels = []
        dic = {}
        for el in es:
            for label in el:
                if label not in dic:
                    edge_labels.append(label)
                    dic[label] = True
        param = self.sample_edge_label_param(n=len(edge_labels))
        self.params = param
        self.param_keys = {edge_labels[idx]:idx for idx in range(len(edge_labels))}
        return None

    def get_edge_labels(self, G):
        """Retrieve all edge labels from a graph

        Args:
            G (nx medi graph): nx multi edge dir. graph
        Returns:
            list with edge labels
        """

        out = []
        for (n1, n2, _) in G.edges:
            edat = G.get_edge_data(n1, n2)
            for k in edat:
                label = edat[k]["label"]
                out.append(label)
        return out

    def sample_edge_label_param(self, n=1):
        """initialize edge parameters. 
        
        The idea with min entropy 
        is to better be able distinguish between edges. This helps with 
        label discirmintation in ARG tasks (but slightly reduces performance
        in other tasks, for other tasks similar or learnt edge weights may 
        be better)

        Args:
            n (int): how many parameters are needed?
        
        Returns:
            array with parameters
        """
        
        params = []

        if self.init == "ones":
            params = np.ones((n, 1))
        
        if self.init == "constant":
            params = np.full((n, 1), 0.2)
        
        elif self.init == "random_uniform":
            for _ in range(n):
                params.append(np.random.uniform(0.2, 0.35, size=(1)))
            params = np.array(params)
        
        elif self.init == "min_entropy": 
            for _ in range(n):
                sample = []
                for _ in range(10):
                    sample.append(np.random.uniform(0.2, 0.35, size=(1)))
                entropies = []
                for i in range(10):
                    entropies.append(
                            entropy(np.array(np.array(params).flatten() 
                                        + list(sample[i])).flatten()))
                argmin = np.argmin(entropies)
                params.append(sample[argmin])
            params = np.array(params)
        
        return params
            
class NodeDistanceMatrixGenerator():

    """Given a list with graph tuples, it generates node embeddings 
    and produces distance matrix"""

    def __init__(self, params=None, param_keys=None, 
                    iters=2, communication_direction="both"):
        """Intitalizes node embedding generatror
            
            Args:
                params (array): edge parameters
                param_keys (dict): maps from edge labels to parameter index
                iters (int): contextualization iterations
                communication_direction: either "both", "fromout", or "fromin"
                                        specifies message passing direction 
                                        (see arguments of main_wlk_wasser.py)
        """
        
        self.params = params
        self.active_params = np.zeros(len(self.params))
        if params is None:
            self.params = []
        self.param_keys = param_keys
        if param_keys is None:
            self.param_keys = {}
            self.unk_edge = 0.2
        else:
            self.unk_edge = np.random.rand(self.params.shape[1]) 
        self.iters = iters
        self.communication_direction = communication_direction
        return None
     
    def _wl_embed_single(self, graphtuple):
        """Embed nodes and generate distance matrix for a (A,B) graph tuple
            This is required as EMD input.
        
        Args:
            graphtuple (tuple): graph tuple (A,B) of nx medi graphs
            iters (int): what degree of contextualization is wanted?
                        default=2

        Returns:
            - distance matrix
            - node weights graph 1
            - node weights graph 2
            - order of nodes in matrix graph1
            - order of nodes in matrix graph2
        """

        a1 = graphtuple[0]
        a2 = graphtuple[1]
        
        e1, order1 = self.collect_graph_embed(a1)
        e2, order2 = self.collect_graph_embed(a2)
        E1, E2 = self._WL_latent(a1, a2, iters=self.iters)
        E1 = np.concatenate([e1, E1], axis=1)
        E2 = np.concatenate([e2, E2], axis=1)
        
        v1, v2, dists = self._get_emd_input(E1, E2)
        
        return dists, v1, v2, order1, order2

    def generate(self, amrs1, amrs2, parallel=False):
        """two (parallel) data sets, call _wl_embed_single 
        on each paired graph
        
        Args:
            amrs1 (list): list with nx medi graphs a_1,...,a_n
            amrs2 (list): list with nx medi graphs b_1,...,b_n
            parallel (boolean): parallelize computation? default=no

        Returns:
            output of _wl_embed_single for each graph pair
        """

        self.active_params = np.zeros(len(self.params))
        assert len(amrs2) == len(amrs1)
         
        zipped = list(zip(amrs1, amrs2))
        data = []

        if parallel:
            with multiprocessing.Pool(10) as p:
                data = p.map(self._wl_embed_single, zipped)
        else:
            for i in range(len(zipped)):
                if i % 100 == 0:
                    logger.info("{}/{} node distance matrices computed".format(i, len(zipped)))
                data.append(self._wl_embed_single(zipped[i]))
        return data

    def _get_emd_input(self, mat1, mat2):
        """Prepares input for pyemd
        
        Args:
            mat1 (matrix): embeddings fror nodes x_1,...,x_n
            mat2 (matrix): embeddings fror nodes y_1,...,y_m

        Returns:
            - prior weights for nodes of A
            - prior weights for nodes of B
            - cost matrix
        """

        mat1 = mat1 / np.linalg.norm(mat1, axis=1)[:,None] 
        mat2 = mat2 / np.linalg.norm(mat2, axis=1)[:,None]
        
        # construct prior weights of nodes... all are set equal here
        v1 = np.concatenate([np.ones(mat1.shape[0]), np.zeros(mat2.shape[0])])
        v2 = np.concatenate([np.zeros(mat1.shape[0]), np.ones(mat2.shape[0])])
        v1 = v1 / sum(v1)
        v2 = v2 / sum(v2)
        
        # build cost matrix
        dist_mat = np.zeros(shape=(len(v1), len(v1)), dtype=np.double)
        
        idxs1 = list(range(0, len(mat1)))
        idxs2 = list(range(len(mat1), len(mat2) + len(mat1)))
        
        #set distances
        dx = cdist(mat1, mat2)
        dist_mat[np.ix_(idxs1, idxs2)] = dx
        
        return v1, v2, dist_mat

    def norm(self, x):
        """scale vector to length 1"""
        div = np.linalg.norm(x)
        return x / div

    def set_params(self, params, idx=None):
        """set edge params"""
        if idx is None:
            self.params = params
        else:
            self.params[idx] = params
        return None
    
    def get_params(self):
        """get edge params"""
        return self.params
     
    def collect_graph_embed(self, nx_latent):
        """collect the node embeddings from a graph
        
        Args:
            nx_latent (nx medi graph): a graph that has node embeddings 
                                        as attributes
        Returns:
            - the node embeddings
            - labels of nodes
        """

        vecs = []
        labels = []
        for node in nx_latent.nodes:
            vecs.append(nx_latent.nodes[node]["latent"])
            labels.append(node)
        return np.array(vecs), labels

    def maybe_has_param(self, label):
        """safe retrieval of an edge parameter"""
        if label not in self.param_keys:
            return self.unk_edge
        self.active_params[self.param_keys[label]] += 1
        return self.params[self.param_keys[label]]

    def _communicate_node(self, G, node):
        """In-place contextualization of a node with neighborhood

        Args:
            G (nx medi graph): input graph
            node: the node

        Returns:
            None
        """

        summ = []
         
        if self.communication_direction in ["both", "fromout"]:
            # iterate over outgoing
            for nb in G.neighbors(node):
                # get node embedding of neighbor
                latentv = G.nodes[nb]["latent"].copy()
                # multi di graph can have multiple edges, so iterate over all
                for k in G.get_edge_data(node, nb):
                    e_latent = self.maybe_has_param(G.get_edge_data(node, nb)[k]['label'])
                    latentv *= e_latent
                summ.append(latentv)
              
        if self.communication_direction in ["both", "fromin"]:
            # iterate over incoming, see above
            for nb in G.predecessors(node):
                latentv = G.nodes[nb]["latent"].copy()
                for k in G.get_edge_data(nb, node):
                    e_latent = self.maybe_has_param(G.get_edge_data(nb, node)[k]['label'])
                    latentv *= e_latent
                summ.append(latentv)
        
        # handle possible exception case
        if not summ:
            summ = np.zeros((1, G.nodes[node]["latent"].shape[0]))

        # compute new embedding of node
        summ = np.mean(summ, axis=0)
        G.nodes[node]["newlatent"] = G.nodes[node]["latent"] + summ
        return None

    def _communicate(self, nx_latent):
        """Applies contextualization (in-place) for all nodes of a graph
        
        Args:
            G (nx medi graph): input graph

        Returns:
            None
        """
        
        #collect new embeddings
        for node in nx_latent.nodes:
            self._communicate_node(nx_latent, node)

        #set new embeddings
        for node in nx_latent.nodes:
            nx_latent.nodes[node]["latent"] = nx_latent.nodes[node]["newlatent"]
        
        return None

    def _wl_iter_latent(self, nx_g1_latent, nx_g2_latent):
        """apply one WL iteration and get node embeddings for two graphs A and B

        Args:
            nx_g1_latent (nx medi graph): graph A
            nx_g2_latent (nx medi graph): graph B

        Returns:
            - contextualized node embeddings for A
            - contextualized node embeddings for B
            - new copy of A
            - new copy of B
        """

        self._communicate(nx_g1_latent)
        self._communicate(nx_g2_latent)
        mat1, _ = self.collect_graph_embed(nx_g1_latent)
        mat2, _ = self.collect_graph_embed(nx_g2_latent)
        return mat1, mat2

    def _WL_latent(self, nx_g1_latent, nx_g2_latent, iters=2):
        """apply K WL iteration and get node embeddings for two graphs A and B

        Args:
            nx_g1_latent (nx medi graph): graph A
            nx_g2_latent (nx medi graph): graph B

        Returns:
            - contextualized node embeddings for A for k=1,..k=n
            - contextualized node embeddings for B for k=1,..k=n
        """

        v1s = []
        v2s = []
        for _ in range(iters):
            x1_mat, x2_mat = self._wl_iter_latent(nx_g1_latent, nx_g2_latent)
            v1s.append(x1_mat)
            v2s.append(x2_mat)
        g_embed1 = np.concatenate(v1s, axis=1)
        g_embed2 = np.concatenate(v2s, axis=1)
        return g_embed1, g_embed2


class EmSimilarity():

    def __init__(self):
        return None
  
    def _ems_single(self, data):
        """Predict WWLK similarity for a (A,B) graph tuple
        
        Args:
            graphtuple (tuple): graph tuple (A,B) of nx medi graphs
            iters (int): what degree of contextualization is desired?
                        default=2

        Returns:
            similarity
        """
        
        dists, v1, v2 = data
        
        #compute wmd
        emd, flow = emd_with_flow(v1, v2, dists)
        
        #change to similarity in -1, 1
        ems = emd * -1
        ems += 1

        return (ems, flow)

    def ems_multi(self, distss, v1s, v2s, parallel=False):
        
        """Predict WWLK similarities for two (parallel) data sets
        
        Args:
            amrs1 (list): list with nx medi graphs a_1,...,a_n
            amrs2 (list): list with nx medi graphs b_1,...,b_n
            parallel (boolean): parallelize computation? default=no

        Returns:
            similarities for AMR graphs
        """
         
        zipped = list(zip(distss, v1s, v2s))
        preds = []

        if parallel:
            with multiprocessing.Pool(10) as p:
                preds = p.map(self._ems_single, zipped)
        else:
            for i in range(len(zipped)):  
                if i % 100 == 0:
                    logger.info("{}/{} Wasserstein distances computed".format(i, len(zipped)))
                preds.append(self._ems_single(zipped[i]))

        return preds


class WasserWLK(GraphSimilarityPredictorAligner):

    def __init__(self, preprocessor, iters=2, stability=0, communication_direction="both"):
        """Initializes Wasserstein Weisfeiler Leman Kernel

        Args:
            preprocessor (Preprocessor): an object that assigns embeddings 
                                        to graph nodes and labels
            iters (int): K
            stability (int): in case there is randomness in pre-processing 
                             (e.g., random embeddings for node labels not found in word2vec)
                            then we compute an expected distance matrix by repeated sampling
            communication_direction (string): communication direction in which messages are passed
        
        Returns:
            None
        """

        self.preprocessor = preprocessor
        self.iters = iters
        self.ems = EmSimilarity()
        self.wl_dist_mat_generator = None
        self.stability = stability
        self.communication_direction = communication_direction

        return None
    
    def _gen_dist_mats(self, graphs1, graphs2):
        
        # if resettable then reset 
        # recall: preprocessor generates random embeddings for OOV 
        # so this is needed for distance matrix samples (stability)
        if self.preprocessor.is_resettable:
            self.preprocessor.reset()
        
        # if preprocessor not fitted, fit prepro and create Wl generator
        if not self.preprocessor.params_ready:
            self.preprocessor.prepare(graphs1, graphs2)
            params = self.preprocessor.params
            param_keys = self.preprocessor.param_keys
            self.wl_dist_mat_generator = NodeDistanceMatrixGenerator(params=params, 
                                                                    param_keys=param_keys, 
                                                                    iters=self.iters,
                                                                    communication_direction=self.communication_direction)

        # no WL generator ready, create WL generator
        if self.wl_dist_mat_generator == None:
            params = self.preprocessor.params
            param_keys = self.preprocessor.param_keys
            self.wl_dist_mat_generator = NodeDistanceMatrixGenerator(params=params, 
                                                                    param_keys=param_keys, 
                                                                    iters=self.iters,
                                                                    communication_direction=self.communication_direction)
        
        # init node embeddings
        self.preprocessor.transform(graphs1, graphs2)
        
        # contextualize
        data = self.wl_dist_mat_generator.generate(graphs1, graphs2)

        return data

    def _gen_expected_dist_mats(self, graphs1, graphs2):
        
        data = self._gen_dist_mats(graphs1, graphs2)
        mats = [dat[0] for dat in data]
         
        n = self.stability #100
        for i in range(n):
            logger.info("sampled {}/{} distance matrices".format(i, n))
            mats_tmp = [dat[0] for dat in self._gen_dist_mats(graphs1, graphs2)]
            mats = [mat + mats_tmp[i] for i, mat in enumerate(mats)]
        new_data = []
        for i, mat in enumerate(mats):
            dat = [mat / (n + 1)] + list(data[i][1:])
            new_data.append(dat)
        return new_data
    

    def _predict_and_align(self, amrs1, amrs2, node2nodeorig_1, node2nodeorig_2):
        """Predict WWLK similarities for two (parallel) data sets 
            and get alignments
        
        Args:
            amrs1 (list): list with nx medi graphs a_1,...,a_n
            amrs2 (list): list with nx medi graphs b_1,...,b_n
            nodemap1 (dict): mapping from nx medi graph nodes 
                             to standard AMR variables for amrs1
            nodemap2 (dict): mapping from nx medi graph nodes 
                             to standard AMR variables for amrs2

        Returns:
            - similarities for AMR graphs
            - alignments
        """
 
        data = self._gen_expected_dist_mats(amrs1, amrs2)
        mats = [dat[0] for dat in data]
        v1s = [dat[1] for dat in data]
        v2s = [dat[2] for dat in data]
        orders1 = [dat[3] for dat in data]
        orders2 = [dat[4] for dat in data]
        sim_flow_list = self.ems.ems_multi(mats, v1s, v2s)
        
        preds = []
        aligns = []
        
        for i in range(len(sim_flow_list)):
            sim, flow = sim_flow_list[i]
            dist_mat = mats[i]
            order1 = orders1[i]
            order2 = orders2[i]
            align_dict = {} 
            # project alignment to orig AMR graphs
            for j, label in enumerate(order1):
                align_dict[node2nodeorig_1[i][label]] = {}
                cutv = len(order1)
                row = flow[j][cutv:]
                cost_row = dist_mat[j][cutv:]
                for k, num in enumerate(row):
                    if num > 0.0:
                        varnode1 = node2nodeorig_1[i][label]
                        varnode2 = node2nodeorig_2[i][order2[k]]
                        align_dict[varnode1][varnode2] = (num, cost_row[k])
            aligns.append(align_dict)
            preds.append(sim)
        return preds, aligns


    def _predict(self, amrs1, amrs2):
        """Predict WWLK similarities for two (parallel) data sets 
            and get alignments
        
        Args:
            amrs1 (list): list with nx medi graphs a_1,...,a_n
            amrs2 (list): list with nx medi graphs b_1,...,b_n
            nodemap1 (dict): mapping from nx medi graph nodes 
                             to standard AMR variables for amrs1
            nodemap2 (dict): mapping from nx medi graph nodes 
                             to standard AMR variables for amrs2

        Returns:
            - similarities for AMR graphs
            - alignments
        """

        
        data = self._gen_expected_dist_mats(amrs1, amrs2)
        mats = [dat[0] for dat in data]
        v1s = [dat[1] for dat in data]
        v2s = [dat[2] for dat in data]
        sim_flow_list = self.ems.ems_multi(mats, v1s, v2s)
        preds = [dat[0] for dat in sim_flow_list]
        return preds


class WLK(GraphSimilarityPredictor):
    
    def __init__(self, simfun='cosine', iters=2, communication_direction="both"):
        self.simfun = simfun
        self.iters = iters
        self.communication_direction = communication_direction

    def _predict(self, amrs1, amrs2):
        """predicts similarity scores for paired graphs
        
        Args:
            amrs1 (list with nx medi graphs): graphs
            amrs2 (list with nx medi graphs): other graphs

        Returns:
            list with floats (similarities)
        """

        kvs = []
        for i, a1 in enumerate(amrs1):
            a2 = amrs2[i]
            gs1 = self.get_stats(a1, a2, stattype='nodeoccurence')
            gs2 = self.get_stats(a1, a2, stattype='tripleoccurence')
            v1, v2 = gs1[0], gs1[1]
            v1, v2 = np.concatenate([v1, gs2[0]]), np.concatenate([v2, gs2[1]])
            kv = self.wlk(a1, a2, iters=self.iters, kt=self.simfun, init_vecs=(v1, v2), 
                            weighting="linear", stattype="nodeoccurence")
            if np.isnan(kv):
                kv = 0.0
            kvs.append(kv)
            if i % 100 == 0:
                logger.info("{}/{} graph pairs fully processed".format(i, len(amrs1)))
        return kvs
        
    def wl_gather_node(self, node, G):
        """ gather edges+labels for a node from the neighborhood
        
        Args:
            node (hashable object): a node of the graph
            G (nx medi graph): the graph

        Returns:
            a list with edge+label from neighbors
        """
        
        newn = [G.nodes[node]["label"]]
        
        if self.communication_direction in ["both", "fromout"]:
            for nb in G.neighbors(node):
                for k in G.get_edge_data(node, nb):
                    el = G.get_edge_data(node, nb)[k]['label']
                    label = G.nodes[nb]["label"]
                    newn.append(el + '_' + label)
        
        if self.communication_direction in ["both", "fromin"]:
            for nb in G.predecessors(node):
                for k in G.get_edge_data(nb, node):
                    el = G.get_edge_data(nb, node)[k]['label']
                    label = G.nodes[nb]["label"]
                    newn.append(el + '_' + label)
        
        return newn
    
    def wl_gather_nodes(self, G):
        """apply gathering (wl_gather_node) for all nodes
        
        Args:
            G (nx medi graph): the graph

        Returns:
            a dictionary node -> neigjborhood
        """

        dic = {}
        for node in G.nodes:
            label = G.nodes[node]["label"]
            dic[label] = self.wl_gather_node(node, G)
        return dic
    
    def sort_relabel(self, dic1, dic2):
        """form aggregate labels via sorting
        
        Args: 
            dic1 (dict): node-neighborhood dict of graph A
            dic2 (dict): node-neighborhood dict of graph B
            
        Returns:
            two dicts where keys are same and values are strings
        """

        for node in dic1:
            dic1[node] = ' ::: '.join(list(sorted(dic1[node])))

        for node in dic2:
            dic2[node] = ' ::: '.join(list(sorted(dic2[node])))

        return dic1, dic2
     
    def wlk(self, nx_g1, nx_g2, iters=2, weighting='linear', kt='dot', 
            stattype='nodecount', init_vecs=(None, None)):
        """compute WL kernel similarity of graph A and B

        Args:
            nx_g1 (nx medi graph): graph A
            nx_g2 (nx medi graph): graph B
            iters (int): iterations
            weighting (string): decrease weight of iteration stats
            kt (string): kernel type, default dot
            stattype (string): which features? default: nodecount
            init_vecs (tuple): perhaps there are already 
                             some features for A and B?
        
        Returns:
            kernel similarity
        """

        v1s, v2s, _ = self.wl(nx_g1, nx_g2, iters=iters, stattype=stattype)
        if init_vecs[0] is not None:
            v1s = [
             init_vecs[0]] + v1s
            v2s = [init_vecs[1]] + v2s
        if weighting == 'exp':
            wts = np.array([np.e ** (-1 * x) for x in range(0, 100)])
            wts = wts[:len(v1s)]
            v1s = [vec * wts[i] for i, vec in enumerate(v1s)]
            v2s = [vec * wts[i] for i, vec in enumerate(v2s)]
        if weighting == 'linear':
            wts = np.array([1 / (1 + x) for x in range(0, 100)])
            wts = wts[:len(v1s)]
            v1s = [vec * wts[i] for i, vec in enumerate(v1s)]
            v2s = [vec * wts[i] for i, vec in enumerate(v2s)]
        v1 = np.concatenate(v1s)
        v2 = np.concatenate(v2s)
        if kt == 'cosine':
            return 1 - cosine(v1, v2)
        if kt == 'rbf':
            gamma = 2.5
            diff = v1 - v2
            dot = -1 * np.einsum('i,i ->', diff, diff)
            div = 2 * gamma ** 2
            return np.exp(dot / div)
        # dot product
        return np.einsum('i,i ->', v1, v2)


    def wl(self, nx_g1, nx_g2, iters=2, stattype='nodecount'):
        """collect vectors over WL iterations

        Args:
            nx_g1 (nx medi graph): graph A
            nx_g2 (nx medi graph): graph B

        Returns:
            a list for every graph that contains vectors
        """

        v1s = []
        v2s = []
        vocabs = []
        for _ in range(iters):
            x1, x2, vocab = self.wl_iter(nx_g1, nx_g2, stattype=stattype)
            v1s.append(x1)
            v2s.append(x2)
            vocabs.append(vocab)
        return v1s, v2s, vocabs

    def wl_iter(self, nx_g1, nx_g2, stattype='nodecount'):
        """collect vectors over one WL iteration

        Args:
            nx_g1 (nx medi graph): graph A
            nx_g2 (nx medi graph): graph B

        Returns:
            - a list for every graph that contains vectors
            - new aggreagate graphs
        """

        dic_g1 = self.wl_gather_nodes(nx_g1)
        dic_g2 = self.wl_gather_nodes(nx_g2)
        d1, d2 = self.sort_relabel(dic_g1, dic_g2)
        self.update_node_labels(nx_g1, d1)
        self.update_node_labels(nx_g2, d2)
        stats1, stats2, vocab = self.get_stats(nx_g1, nx_g2, stattype=stattype)
        return stats1, stats2, vocab

    def update_node_labels(self, G, dic):
        for node in G.nodes:
            label = G.nodes[node]["label"]
            G.nodes[node]["label"] = dic[label]
        return None


    
    def get_stats(self, g1, g2, stattype='nodecount'):
        """get feature vec for a statistitic type
        
        Args: 
            g1 (nx medi graph): graph A
            g2 (nx medi graph): graph B
            stattype (string): statistics type, default: node count

        Returns:
            - vector for A
            - vector for B
            - vocab
        """
        
        vec1, vec2, vocab = None, None, None
        if stattype == 'nodecount':
            vec1, vec2, vocab =  self.nc(g1, g2)
        if stattype == 'nodeoccurence':
            v1, v2, voc = self.nc(g1, g2)
            v1[v1>1] = 1
            v2[v2>1] = 1
            vec1, vec2, vocab =  v1, v2, voc
        if stattype == 'triplecount':
            vec1, vec2, vocab = self.tc(g1, g2)
        if stattype == 'tripleoccurence':
            v1, v2, voc = self.tc(g1, g2)
            v1[v1>1] = 1
            v2[v2>1] = 1
            vec1, vec2, vocab =  v1, v2, voc
        
        return vec1, vec2, vocab
    
    def create_fea_vec(self, items, vocab):
        """create freture vector from bow list and vocab
        
        Args:
            items (list): list with items e.g. [x, y, z]
            vocab (dict): dict with item-> id eg. {x:2, y:4, z:5}

        Returns:
            feature vector, e.g., [0, 0, 1, 0, 1, 1]
        """

        vec = np.zeros(len(vocab))
        for item in items:
            vec[vocab[item]] += 1
        return vec

    def nc(self, g1, g2):
        """ feature vector constructor for node BOW of two graphs
        
        Args:
            g1 (nx medi graph): graph A
            g2 (nx medi graph): graph B
        
        Returns:
            feature vector for graph A, feature vector for graph B, vocab 
        """
        
        vocab = {}
        i = 0
        g1bow = []
        g2bow = []
        for node in g1.nodes:
            label = g1.nodes[node]["label"]
            g1bow.append(label)
            if label not in vocab:
                vocab[label] = i
                i += 1
        for node in g2.nodes:
            label = g2.nodes[node]["label"]
            g2bow.append(label)
            if label not in vocab:
                vocab[label] = i
                i += 1

        vec1 = self.create_fea_vec(g1bow, vocab)
        vec2 = self.create_fea_vec(g2bow, vocab)
        return (vec1, vec2, vocab)

    def tc(self, g1, g2):
        """ feature vector constructor for triple BOW of two graphs
        
        Args:
            g1 (nx medi graph): graph A
            g2 (nx medi graph): graph B
        
        Returns:
            feature vector for graph A, feature vector for graph B, vocab 
        """

        vocab = {}
        g1bow = []
        g2bow = []
        i = 0
        for node in g1.nodes:
            for nb in g1.neighbors(node):
                for k in g1.get_edge_data(node, nb):
                    el = g1.nodes[node]["label"]
                    el += '_' + g1.get_edge_data(node, nb)[k]['label'] 
                    el += '_' + g1.nodes[nb]["label"]
                    g1bow.append(el)
                    if el not in vocab:
                        vocab[el] = i
                        i += 1
        for node in g2.nodes:
            for nb in g2.neighbors(node):
                for k in g2.get_edge_data(node, nb):
                    el = g2.nodes[node]["label"]
                    el += '_' + g2.get_edge_data(node, nb)[k]['label'] 
                    el += '_' + g2.nodes[nb]["label"]
                    g2bow.append(el)
                    if el not in vocab:
                        vocab[el] = i
                        i += 1
 
        vec1 = self.create_fea_vec(g1bow, vocab)
        vec2 = self.create_fea_vec(g2bow, vocab)
        vec1 = vec1
        vec2 = vec2
        return vec1, vec2, vocab
