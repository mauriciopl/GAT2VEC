from __future__ import print_function

from deepwalk import graph
from GAT2VEC import parsers, paths
from gensim.models import Word2Vec
import random


class Gat2Vec(object):
    """
    GAT2VEC learns an embedding jointly from structural contexts and attribute contexts
    employing a single layer of neural network.
    """

    def __init__(self, structural_graph, attribute_graph):
        print("Initializing gat2vec")
        self._seed = 1
        self.Gs = structural_graph
        self.Ga = attribute_graph

    def _filter_walks(self, walks, node_num):
        """ filter attribute nodes from walks in attributed graph."""
        filter_walks = []
        for walk in walks:
            if int(walk[0]) <= node_num:
                fwalks = [nid for nid in walk if int(nid) <= node_num]
                filter_walks.append(fwalks)
        return filter_walks

    def _train_word2Vec(self, walks, dimension_size, window_size, cores):
        """ Trains jointly attribute contexts and structural contexts."""
        print("Learning Representation")
        return Word2Vec(
            [list(map(str, walk)) for walk in walks],
            size=dimension_size,
            window=window_size, min_count=0, sg=1,
            workers=cores
        )

    def train_gat2vec(self, nwalks, wlength, dsize, wsize):
        print("Running the basic Gat2Vec algorithm")
        return self._train_gat2vec(dsize, nwalks, wlength, wsize)

    def train_labelled_gat2vec(self, nwalks, wlength, dsize, wsize, dataset_dir, TR):
        """ Trains on labelled dataset, i.e class labels are used as an attribute """
        print("Training on Labelled Data")
        for tr in TR:
            f_ext = "label_" + str(int(tr * 100)) + '_na'
            self.Ga = parsers.get_graph(dataset_dir, f_ext)
            gat2vec_model = self._train_gat2vec(dsize, nwalks, wlength, wsize)
        return gat2vec_model  # TODO: it used to save this, change and return a dict etc.

    def train_gat2vec_bip(self, nwalks, wlength, dsize, wsize):
        """ Trains on the bipartite graph only"""
        print("Learning Representation on Bipartite Graph")
        return self._train_gat2vec(dsize, nwalks, wlength, wsize, add_structure=False)

    def _train_gat2vec(self, dsize, nwalks, wlength, wsize, add_structure=True):
        print("Random Walks on Structural Graph")
        walks_structure = graph.build_deepwalk_corpus(
            self.Gs,
            num_paths=nwalks,
            path_length=wlength,
            alpha=0,
            rand=random.Random(self._seed)
        )
        print("Random Walks on Attribute Graph")
        walks_attribute = graph.build_deepwalk_corpus(
            self.Ga,
            num_paths=nwalks,
            path_length=wlength * 2,
            alpha=0,
            rand=random.Random(self._seed)
        )
        walks = self._filter_walks(walks_attribute, len(self.Gs.nodes()))

        if add_structure:
            walks = walks_structure + walks

        gat2vec_model = self._train_word2Vec(walks, dsize, wsize, 4)
        return gat2vec_model
