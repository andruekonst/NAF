import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import NamedTuple, Optional, Tuple
from .forests import *
from time import time
from sklearn.preprocessing import OneHotEncoder


class ClfRegHot:
    @abstractmethod
    def fit(self, X, y) -> 'ClfRegHot':
        pass

    @abstractmethod
    def optimize_weights(self, X, y) -> 'ClfRegHot':
        """Optimize weights for the new dataset."""
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        pass

    @abstractmethod
    def predict_original(self, X) -> np.ndarray:
        pass


def _convert_labels_to_probas(y, encoder=None):
    if y.ndim == 2 and y.shape[1] >= 2:
        return y, encoder
    if encoder is None:
        encoder = OneHotEncoder()
        y = encoder.fit_transform(y.reshape((-1, 1)))
    else:
        y = encoder.transform(y.reshape((-1, 1)))
    return y, encoder


class AttentionForest(ClfRegHot):
    def __init__(self, params):
        self.params = params
        self.forest = None
        self._after_init()

    def _after_init(self):
        self.onehot_encoder = None

    def _preprocess_target(self, y):
        if self.params.task == TaskType.CLASSIFICATION:
            y, self.onehot_encoder = _convert_labels_to_probas(y, self.onehot_encoder)
        return y


    def fit(self, X, y) -> 'AttentionForest':
        forest_cls = FORESTS[ForestType(self.params.kind, self.params.task)]
        self.forest = forest_cls(**self.params.forest)
        print("Start fitting Random forest")
        start_time = time()
        self.forest.fit(X, y)
        end_time = time()
        print("Random forest fit time:", end_time - start_time)
        return self

    def optimize_weights(self, X, y_orig) -> 'AttentionForest':
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        raise NotImplementedError()

    def predict_original(self, X):
        if self.params.task == TaskType.REGRESSION:
            return self.forest.predict(X)
        elif self.params.task == TaskType.CLASSIFICATION:
            return self.forest.predict_proba(X)
        raise ValueError(f'Unsupported task type in predict_original: "{self.params.task}"')

