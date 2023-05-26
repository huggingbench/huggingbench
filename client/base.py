# This file contains re-usable code and utility classes for the client
import threading
import logging
from typing import Union
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset

LOG = logging.getLogger(__name__)


DatasetAlias = Union[DatasetDict, Dataset,
                     IterableDatasetDict, IterableDataset]


class DatasetProvider:
    def get_dataset(self) -> DatasetAlias:
        """Return a dataset"""
        pass

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
