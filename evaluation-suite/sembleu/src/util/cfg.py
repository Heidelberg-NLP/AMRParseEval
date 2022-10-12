import itertools
from util import log
from util.logarithm import logadd, logsum, LOGZERO
from collections import defaultdict
from queue import PriorityQueue
import math
import heapq

class NonterminalLabel(object):
    """
    There can be multiple nonterminal edges with the same symbol. Wrap the
    edge into an object so two edges do not compare equal.
    Nonterminal edges carry a nonterminal symbol and an index that identifies
    it uniquely in a rule.
    """

    #label_matcher = re.compile("(?P<label>.*?)(\[(?P<index>.*)\])?$")

    def __init__(self, label, index = None):
        # Parsing and setting the sychronization index is now handled
        # by the grammar parser

        #if index is not None:
        self.label = label
        self.index = index
        #else:
        #    match = NonterminalLabel.label_matcher.match(label)
        #    self.index = match.group("index")
        #    self.label = match.group("label")

    def __eq__(self, other):
        try:
            return self.label == other.label and self.index == other.index
        except AttributeError:
            return False

    def __repr__(self):
        return "NT(%s)" % str(self)

    def __str__(self):
        if self.index is not None:
            return "%s$%s" % (str(self.label), str(self.index))
        else:
            return "%s$" % str(self.label)

    def __hash__(self):
        return 83 * hash(self.label) + 17 * hash(self.index)

    @classmethod
    def from_string(cls, s):
        label, index = s.split("$")
        return NonterminalLabel(label, index or None)


class Chart(dict):
    """
    A CKY style parse chart that can return k-best derivations and can return inside and outside probabilities.
    """

    def kbest(self, item, k):
        """
        Return the k-best derivations from this chart.
        """

        if item == "START":
            rprob = 0.0
        else:
            rprob = item.rule.weight

        # If item is a leaf, just return it and its probability
        if not item in self:
            if item == "START":
                log.info("No derivations.")
                return []
            else:
                return [(rprob, item)]

        pool = []
        # Find the k-best options for this rule.
        # Compute k-best for each possible split and add to pool.
        for split in self[item]:
            nts, children = list(zip(*list(split.items())))
            kbest_each_child = [self.kbest(child, k) for child in children]

            all_combinations  = list(itertools.product(*kbest_each_child))

            combinations_for_sorting = []
            for combination in all_combinations:
                 weights, trees = list(zip(*combination))
                 heapq.heappush(combinations_for_sorting,(sum(weights)+rprob,trees))

            #combinations_for_sorting.sort(reverse=True)

            for prob, trees in heapq.nlargest(k, combinations_for_sorting):
                new_tree = (item, dict(list(zip(nts, trees))))
                heapq.heappush(pool, (prob, new_tree))

        # Return k-best from pool
        return heapq.nlargest(k, pool) #sorted(pool, reverse=True)[:k]

    ### Methods for Inside/Outside

    def inside_scores(self):
        inside_probs = {}

        def compute_scores(chart, item):
            """
            Here we compute the inside scores for each rule and split, i.e. the
            sum of all possible ways to decompose this item.
            This is the inside computation of the inside-outside algorithm
            for wRTGs described in Graehl&Knight 2004 "Training tree transducers".
            """

            if item == "START":
                weight = 0.0
            else:
                weight = item.rule.weight
            if item in self:
                beta_each_split = []
                for split in self[item]:
                    nts, children = list(zip(*list(split.items())))
                    beta_each_child = [compute_scores(chart, child) for child in children]
                    beta_this_split = sum(beta_each_child)
                    beta_each_split.append(beta_this_split)
                beta_this_item = weight + logsum(beta_each_split)

            else: # Leaf case
                beta_this_item = weight

            inside_probs[item] = beta_this_item
            return beta_this_item

        beta_start = compute_scores(self, "START")
        return inside_probs


    def outside_scores(self, inside_probs):
        outside_probs = {}
        outside_probs["START"] = 0.0

        def compute_scores(chart, item):
            """
            Here we compute the outside scores for each rule and split, i.e. the
            sum of all possible trees that contain this item but do not decompose it.
            This is the outside computation of the inside-outside algorithm
            for wRTGs described in Graehl&Knight 2004 "Training tree transducers".
            """
            if item in chart:
                for split in chart[item]:
                    nts, children = list(zip(*list(split.items())))

                    # An item may be part of multiple splits, so we need to add the outside score when we
                    #  encounter it a second time.
                    for child in children:
                        inside_for_siblings = [inside_probs[c] for c in children if c!=child]
                        alpha_for_child = outside_probs[item] + sum(inside_for_siblings) + child.rule.weight
                        if child in outside_probs:
                            outside_probs[child] = logadd(outside_probs[child],alpha_for_child)
                        else:
                            outside_probs[child] = alpha_for_child
                        compute_scores(chart, child)

        alpha_start = compute_scores(self, "START")
        return outside_probs

    def expected_rule_counts(self, inside_probs, outside_probs):
        counts = defaultdict(float)

        beta_sentence = inside_probs["START"]

        for item in self:
            for split in self[item]:
                nts, children = list(zip(*list(split.items())))
                for child in children:
                    childgamma = outside_probs[item] + inside_probs[child] + child.rule.weight
                    counts[child.rule.rule_id] = logadd(counts[child.rule.rule_id] ,(childgamma - beta_sentence))
        return counts

