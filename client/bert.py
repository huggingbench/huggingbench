# Description: This file contains the code for the BERT model.
import logging
from datasets import load_dataset
from transformers import AutoTokenizer
from base import DatasetAlias

MODEL_NAME = "bert-base-uncased"
MODEL_VERSION = "1"

DATASET_NAME = "squad"
DATASET_COLUMN_NAME1 = "question"
DATASET_COLUMN_NAME2 = "context"

log = logging.getLogger(__name__)


class BertDataset:

    def __init__(self, dataset: str) -> None:
        self.dataset_name = dataset
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        max_sequence_length = getattr(
            tokenizer, "model_max_length", None)
        dataset = load_dataset(self.dataset_name, split="train")

        def preprocess_function(dataset):
            return tokenizer(dataset[DATASET_COLUMN_NAME1], dataset[DATASET_COLUMN_NAME2], truncation=True,
                             padding='max_length', max_length=max_sequence_length, return_tensors="np")

        self.dataset = dataset.with_transform(preprocess_function)
        log.info("Loaded dataset has %d samples", len(self.dataset))

    def dataset(self) -> DatasetAlias:
        return self.dataset
