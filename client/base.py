# This file contains re-usable code and utility classes for the client
import threading
import logging
from typing import Union
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
import numpy as np
from hugging_bench.hugging_bench_config import Input

LOG = logging.getLogger(__name__)


DatasetAlias = Union[DatasetDict, Dataset,
                     IterableDatasetDict, IterableDataset]


class DatasetProvider:
    def get_dataset(self) -> DatasetAlias:
        """Return a dataset"""
        return self.dataset

class DatasetIterator:
    # This is a iterator around a dataset that makes it infinitive and thread-safe
    # We don't want to be recreating the dataset so we just re-start from the beginning
    # once we reach the end
    _lock = threading.Lock()

    def __init__(self, dataset: DatasetAlias, infinite: bool = True):
        self.dataset = dataset
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            if self.index >= len(self.dataset):
                self.index = 0
            item = self.dataset[self.index]
            self.index += 1
            return item

    def __len__(self):
        with self._lock:
            return len(self.dataset)


class UserContext:
    def __init__(self, inf_dataset: DatasetIterator, model_name: str, model_version: str, batch_size: int = 1):
        LOG.info("Loaded dataset with %d samples", len(inf_dataset))
        self.dataset = inf_dataset
        self.model_name = model_name
        self.model_version = model_version
        self.batch_size = batch_size


class DatasetGen(DatasetProvider):
    """ Generates a dataset of random tensors.""" 
    TYPE_MAP = {
    'INT64': np.int64,
    'INT32': np.int32,
    'INT16': np.int16,
    'FP16': np.float16,
    'FP32': np.float32,
    'FP64': np.float64,
}

    def __init__(self, inputs: list[Input], size: int = 100):
        self.inputs = inputs
        self.dataset = dict()
        for i in range(size):
            self.dataset[i] ={input.name: self.radnom_tensor(tuple(input.dims), input.dtype) for input in inputs}

    def radnom_tensor(self, dimensions_tpl, data_type):
        if data_type not in DatasetGen.TYPE_MAP:
            raise ValueError(f"Unsupported data_type {data_type}")
        return np.random.rand(*dimensions_tpl).astype(DatasetGen.TYPE_MAP[data_type])

    def get_dataset(self) -> DatasetAlias:
        return self.dataset
