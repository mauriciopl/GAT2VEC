# -*- coding: utf-8 -*-

from collections import defaultdict, Counter
import logging
import multiprocessing as mp

import numpy as np
import pandas as pd
from sklearn import linear_model, svm, preprocessing
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score, accuracy_score, f1_score, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, ShuffleSplit, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier

from .. import parsers, paths
from ..constants import NESTED_CV_PARAMETERS

__all__ = ['Classification']

logger = logging.getLogger(__name__)


class Classification:
    """This class performs multi-class/multi-label classification tasks."""

    def __init__(self, dataset_dir, output_dir, tr, multilabel=False):
        self.dataset = paths.get_dataset_name(dataset_dir)
        self.output = {"TR": [], "accuracy": [], "f1micro": [], "f1macro": [], "auc": [], "aps": []}
        self.TR = tr  # the training ratio for classifier
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.multi_label = multilabel
        if self.multi_label:
            self.labels, self.label_ind, self.label_count = parsers.get_multilabels(
                self.dataset_dir)
            self.labels = self.binarize_labels(self.labels)
        else:
            self.labels, self.label_ind, self.label_count, self.weights = parsers.get_labels(self.dataset_dir)

    def binarize_labels(self, labels, nclasses=None):
        """ returns the multilabelbinarizer object"""
        if nclasses is None:
            mlb = preprocessing.MultiLabelBinarizer()
            return mlb.fit_transform(labels)
        # for fit_and_predict to return binarized object of predicted classes
        mlb = preprocessing.MultiLabelBinarizer(classes=range(nclasses))
        return mlb.fit_transform(labels)

    def evaluate(self, model, label=False, evaluation_scheme="tr", class_weights=None):
        embedding = 0

        if not label:
            embedding = parsers.get_embeddingDF(model)

        results = dict()
        if evaluation_scheme in ["cv", "svm"]:
            clf = self.get_classifier(evaluation_scheme, class_weight=class_weights)
            results = self.evaluate_cv(clf, embedding, 5)
        elif evaluation_scheme.startswith("nested_"):
            clf = self.get_classifier(evaluation_scheme[7:], class_weight=class_weights)
            results = self.evaluate_nested_cv(clf, embedding, 5)
        elif evaluation_scheme == "tr" or label:
            clf = self.get_classifier('lr', class_weight=class_weights)
            results = defaultdict(list)
            for tr in self.TR:
                logger.debug("TR ... %s", tr)
                if label:
                    model = paths.get_embedding_path_wl(self.dataset_dir, self.output_dir, tr)
                    if isinstance(model, str):
                        embedding = parsers.get_embeddingDF(model)
                results.update(self.evaluate_tr(clf, embedding, tr))

        logger.debug("Training Finished")

        df = pd.DataFrame(results)
        return df.groupby(axis=0, by="TR").mean()

    def get_classifier(self, model, class_weight=None):
        """Returns the classifier, based on the approach."""
        if model in ['lr', 'cv']:
            clf = linear_model.LogisticRegression(solver='lbfgs', class_weight=class_weight)
        elif model == 'svm':
            # https://stackoverflow.com/questions/45384185/what-is-the-difference-between-linearsvc-and-svckernel-linear/45390526
            clf = svm.SVC(kernel='linear', probability=True, class_weight=class_weight)
            # clf = svm.LinearSVC(class_weight=class_weight)  # TODO This is non-probabilistic
        else:
            raise ValueError(f'{model} is not a valid value. Use "cv" for linear regression '
                             f'or "svm" for support vector machine.')
        return clf

    def evaluate_tr(self, clf, embedding, tr):
        """ evaluates an embedding for classification on training ration of tr."""
        ss = ShuffleSplit(n_splits=10, train_size=tr, random_state=2)
        for train_idx, test_idx in ss.split(self.labels):
            X_train, X_test, Y_train, Y_test = self._get_split(embedding, test_idx, train_idx)
            pred, probs = self.get_predictions(clf, X_train, X_test, Y_train, Y_test)
            self._assemble_results(Y_test, tr, pred, probs[:, 1], self.output)
        return self.output

    def evaluate_cv(self, clf, embedding, n_splits):
        """Do a repeated stratified cross validation.

        :param clf: Classifier object.
        :param embedding: The feature matrix.
        :param n_splits: Number of folds.
        :return: Dictionary containing numerical results of the classification.
        """
        embedding = embedding[self.label_ind, :]
        results = defaultdict(list)
        for i in range(10):
            rskf = StratifiedKFold(n_splits=n_splits, shuffle=True)
            for train_idx, test_idx in rskf.split(embedding, self.labels):
                x_train, x_test, y_train, y_test, w_train = self._get_split(embedding, test_idx, train_idx)
                pred, probs = self.get_predictions(
                    clf,
                    x_train,
                    x_test,
                    y_train,
                    y_test,
                    sample_weights=w_train
                )
                self._assemble_results(y_test, i, pred, probs[:, 1], results)
        return results

    def evaluate_nested_cv(self, clf: BaseEstimator, embedding, n_splits):
        """Do a nested cross validation for parameter optimization.

        :param clf: Classifier object.
        :param embedding: The feature matrix.
        :param n_splits: Number of folds.
        :return: Dictionary containing numerical results of the classification.
        """
        pool = mp.Pool(mp.cpu_count())
        results = defaultdict(list)
        embedding = embedding[self.label_ind, :]
        results_iter = [
            pool.apply(self._nested_cross_validation, args=(clf, embedding, n_splits))
            for _
            in range(10)
        ]
        pool.close()
        pool.join()
        for i, (y_test, pred, probs) in enumerate(results_iter):
            self._assemble_results(y_test, i, pred, probs[:, 1], results)
        return results

    def evaluate_nested_cv_bkp(self, clf: BaseEstimator, embedding, n_splits):
        """Do a nested cross validation for parameter optimization.

        :param clf: Classifier object.
        :param embedding: The feature matrix.
        :param n_splits: Number of folds.
        :return: Dictionary containing numerical results of the classification.
        """
        roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)
        grid_search = GridSearchCV(clf, NESTED_CV_PARAMETERS, scoring=roc_auc_scorer, cv=n_splits)
        embedding = embedding[self.label_ind, :]
        best_params = Counter()
        results = defaultdict(list)
        for i in range(10):
            rskf = StratifiedKFold(n_splits=n_splits, shuffle=True)
            for train_idx, test_idx in rskf.split(embedding, self.labels):
                x_train, x_test, y_train, y_test, w_train = self._get_split(embedding, test_idx, train_idx)
                grid_search.fit(x_train, y_train, sample_weight=w_train)
                clf.C = grid_search.best_params_['C']
                clf.class_weight = grid_search.best_params_['class_weight']

                cw_idx = NESTED_CV_PARAMETERS['class_weight'].index(clf.class_weight)
                best_params[('C', clf.C)] += 1
                best_params[('cw', cw_idx)] += 1
                best_params[(clf.C, cw_idx)] += 1

                pred, probs = self.get_predictions(
                    clf,
                    x_train,
                    x_test,
                    y_train,
                    y_test,
                    sample_weights=w_train
                )
                self._assemble_results(y_test, i, pred, probs[:, 1], results)
        logger.info('Best parameters for nested cross validation are:')
        logger.info(best_params)
        return results

    def _nested_cross_validation(self, clf, embedding, n_splits):
        roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)
        grid_search = GridSearchCV(clf, NESTED_CV_PARAMETERS, scoring=roc_auc_scorer, cv=n_splits)
        rskf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        for train_idx, test_idx in rskf.split(embedding, self.labels):
            x_train, x_test, y_train, y_test, w_train = self._get_split(embedding, test_idx, train_idx)
            grid_search.fit(x_train, y_train, sample_weight=w_train)
            clf.C = grid_search.best_params_['C']
            clf.class_weight = grid_search.best_params_['class_weight']

            pred, probs = self.get_predictions(
                clf,
                x_train,
                x_test,
                y_train,
                y_test,
                sample_weights=w_train
            )
            return y_test, pred, probs

    def _assemble_results(self, Y_test, i, pred, probs, results):
        results["TR"].append(i)
        results["accuracy"].append(accuracy_score(Y_test, pred))
        results["f1micro"].append(f1_score(Y_test, pred, average='micro'))
        results["f1macro"].append(f1_score(Y_test, pred, average='macro'))
        if self.label_count == 2:
            results["auc"].append(roc_auc_score(Y_test, probs))
            results["aps"].append(average_precision_score(Y_test, probs))
        else:
            results["auc"].append(0)
            results["aps"].append(0)

    def get_prediction_probs_for_entire_set(self, model):
        embedding = parsers.get_embeddingDF(model)
        embedding = embedding[self.label_ind, :]

        log_reg = linear_model.LogisticRegression(solver='lbfgs')
        clf = OneVsRestClassifier(log_reg)

        clf.fit(embedding, self.labels)  # for multi-class classification
        probs = clf.predict_proba(embedding)
        logger.debug('ROC: %.2f', roc_auc_score(self.labels, probs[:, 1]))

        return probs

    def _get_split(self, embedding, test_id, train_id):
        """Splits the embedding and the labels into training data and test data.

        :param embedding: The embedding calculated for the nodes.
        :param test_id:
        :param train_id:
        :return:
        """
        if self.weights is not None:
            return (
                embedding[train_id], embedding[test_id], self.labels[train_id], self.labels[test_id],
                [self.weights[t_id] for t_id in train_id]
            )
        else:
            return (
                embedding[train_id], embedding[test_id], self.labels[train_id], self.labels[test_id],
                None
            )

    def get_predictions(self, clf, X_train, X_test, Y_train, Y_test, sample_weights=None):
        """Obtains the prediction results based on the chosen classifier and the data.

        :param clf: The classifier.
        :param X_train: Samples used as training data.
        :param X_test: Samples used as test data.
        :param Y_train: Labels used as training data.
        :param Y_test: Labels used as test data.
        :param sample_weights: Sample weights for weighted calculations.
        :return: Tuple with an array with predicted labels and a matrix with
        probabilities for each class.
        """
        if self.multi_label:
            return self.fit_and_predict_multilabel(clf, X_train, X_test, Y_train, Y_test)
        else:
            clf.fit(X_train, Y_train, sample_weight=sample_weights)  # for multi-class classification
            return clf.predict(X_test), clf.predict_proba(X_test)

    def fit_and_predict_multilabel(self, clf, X_train, X_test, y_train, y_test):
        """ predicts and returns the top k labels for multi-label classification
        k depends on the number of labels in y_test."""
        clf.fit(X_train, y_train)
        y_pred_probs = clf.predict_proba(X_test)

        pred_labels = []
        nclasses = y_test.shape[1]
        top_k_labels = [np.nonzero(label)[0].tolist() for label in y_test]
        for i in range(len(y_test)):
            k = len(top_k_labels[i])
            probs_ = y_pred_probs[i, :]
            labels_ = tuple(np.argsort(probs_).tolist()[-k:])
            pred_labels.append(labels_)
        return self.binarize_labels(pred_labels, nclasses), y_pred_probs
