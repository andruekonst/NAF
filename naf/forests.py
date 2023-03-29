import numpy as np
from sklearn.ensemble import (
        RandomForestRegressor,
        RandomForestClassifier,
        ExtraTreesRegressor,
        ExtraTreesClassifier,
)
from enum import Enum, auto
from typing import NamedTuple


class ForestKind(Enum):
    RANDOM = auto()
    EXTRA = auto()

    def need_add_init(self) -> bool:
        return False

    @staticmethod
    def from_name(name: str) -> 'ForestKind':
        if name == 'random':
            return ForestKind.RANDOM
        elif name == 'extra':
            return ForestKind.EXTRA
        else:
            raise ValueError('Wrong forest kind: "{name}".')


class TaskType(Enum):
    CLASSIFICATION = auto()
    REGRESSION = auto()


class ForestType(NamedTuple):
    kind: ForestKind
    task: TaskType


FORESTS = {
    ForestType(ForestKind.RANDOM, TaskType.REGRESSION): RandomForestRegressor,
    ForestType(ForestKind.RANDOM, TaskType.CLASSIFICATION): RandomForestClassifier,
    ForestType(ForestKind.EXTRA, TaskType.REGRESSION): ExtraTreesRegressor,
    ForestType(ForestKind.EXTRA, TaskType.CLASSIFICATION): ExtraTreesClassifier,
}

