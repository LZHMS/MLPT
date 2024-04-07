from dassl.data.datasets import DatasetBase
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json
from dassl.utils import listdir_nohidden
from collections import defaultdict

import random


class COOPDatasetBase(DatasetBase): 
    def __init__(self, train_x=None, train_u=None, val=None, test=None):
        super().__init__(train_x=train_x, train_u=train_u, val=val, test=test)

    def make_noise(self, few_train, num_fp):
        num_classes = self.get_num_classes(few_train)
        for id, item in enumerate(few_train):
            if id >= num_fp:
                break
            item._label = random.randint(0, num_classes - 1)
        return few_train
