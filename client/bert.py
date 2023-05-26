# Description: This file contains the code for the BERT model.
import logging
from datasets import load_dataset
from transformers import AutoTokenizer
from client.base import DatasetAlias, DatasetProvider

MODEL_NAME = "bert-base-uncased"
MODEL_VERSION = "1"

DATASET_NAME = "squad"
DATASET_COLUMN_NAME1 = "question"
DATASET_COLUMN_NAME2 = "context"

log = logging.getLogger(__name__)

class BertDataset(DatasetProvider):

    def __init__(self, dataset: str = DATASET_NAME) -> None:
        self.dataset_name = dataset
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        max_sequence_length = getattr(
            tokenizer, "model_max_length", None)
        dataset = load_dataset(self.dataset_name, split="train")

        def preprocess_function(dataset):
            ### Tokenize contexts and questions (as pairs of inputs)
            ### We are padding to max length to support batching of inputs
            ### Note that padding to `longest` didn't work for batching?! 
            return tokenizer(dataset[DATASET_COLUMN_NAME1], dataset[DATASET_COLUMN_NAME2], truncation=True,
                             padding='max_length', max_length=max_sequence_length, return_tensors="np")

        self.dataset = dataset.with_transform(preprocess_function)
        log.info("Loaded dataset has %d samples", len(self.dataset))

    def get_dataset(self) -> DatasetAlias:
        return self.dataset
