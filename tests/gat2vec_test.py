import os
import unittest

from GAT2VEC.evaluation.classification import Classification
from GAT2VEC.gat2vec import Gat2Vec
from GAT2VEC import paths, parsers
import numpy as np
from tests import constants


class Gat2VecTest(unittest.TestCase):

    def test_train_gat2vec(self):
        if not os.path.isdir(constants.OUTPUT_DIR):
            os.makedirs(constants.OUTPUT_DIR)

        print("loading structural graph")
        structural_graph = parsers.get_graph(constants.DATASET_DIR)
        attribute_graph = parsers.get_graph(constants.DATASET_DIR, 'na')
        g2v = Gat2Vec(structural_graph, attribute_graph)
        model = g2v.train_gat2vec(constants.NUM_WALKS, constants.WALK_LENGTH, constants.DIMENSION,
                                  constants.WINDOW_SIZE)

        fname = ''  # TODO
        model.wv.save_word2vec_format(fname)
        model = fname

        clf_model = Classification(constants.DATASET_DIR, constants.OUTPUT_DIR, tr=constants.TR)
        results_model = clf_model.evaluate(model, label=False, evaluation_scheme="tr")

        ground_truth = paths.get_embedding_path(constants.DATASET_DIR, constants.RESOURCE_DIR)
        results_ground = clf_model.evaluate(ground_truth, label=False, evaluation_scheme="tr")

        np.testing.assert_almost_equal(results_model.values, results_ground.values, decimal=2)

    def test_train_labelled_gat2vec(self):
        if not os.path.isdir(constants.OUTPUT_DIR):
            os.makedirs(constants.OUTPUT_DIR)

        g2v = Gat2Vec(constants.DATASET_DIR, constants.OUTPUT_DIR, is_labelled=True, tr=constants.TR)
        model = g2v.train_gat2vec(constants.NUM_WALKS, constants.WALK_LENGTH, constants.DIMENSION,
                                  constants.WINDOW_SIZE, output=constants.SAVE_OUTPUT)

        clf_model = Classification(constants.DATASET_DIR, constants.OUTPUT_DIR, tr=constants.TR)
        results_model = clf_model.evaluate(model, label=True, evaluation_scheme="tr")

        clf_model.output_dir = constants.RESOURCE_DIR
        print(clf_model.output_dir)
        results_ground = clf_model.evaluate(None, label=True, evaluation_scheme="tr")

        np.testing.assert_almost_equal(results_model.values, results_ground.values, decimal=2)

    def test_train_gat2vec_bip(self):
        if not os.path.isdir(constants.OUTPUT_DIR):
            os.makedirs(constants.OUTPUT_DIR)

        g2v = Gat2Vec(constants.DATASET_DIR, constants.OUTPUT_DIR, is_labelled=False, tr=constants.TR)
        model = g2v.train_gat2vec_bip(constants.NUM_WALKS, constants.WALK_LENGTH,
                                      constants.DIMENSION, constants.WINDOW_SIZE,
                                      output=constants.SAVE_OUTPUT)

        clf_model = Classification(constants.DATASET_DIR, constants.OUTPUT_DIR, tr=constants.TR)
        results_model = clf_model.evaluate(model, label=False, evaluation_scheme="tr")

        ground_truth = paths.get_embedding_path_bip(constants.DATASET_DIR, constants.RESOURCE_DIR)
        results_ground = clf_model.evaluate(ground_truth, label=False, evaluation_scheme="tr")

        np.testing.assert_almost_equal(results_model.values, results_ground.values, decimal=2)

    def test_train_multilabelled_gat2vec(self):
        pass
