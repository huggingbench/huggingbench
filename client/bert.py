# Description: This file contains the code for the BERT model.
import logging
from datasets import load_dataset
from transformers import AutoTokenizer
from base import DatasetAlias, UserContext, InfDataset
from client.triton_user import TritonUser

MODEL_NAME = "bert-base-uncased"
MODEL_VERSION = "1"

DATASET_NAME = "squad"
DATASET_COLUMN_NAME1 = "question"
DATASET_COLUMN_NAME2 = "context"

log = logging.getLogger(__name__)


def _get_dataset(dataset_name: str, column1: str, column2: str) -> DatasetAlias:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    max_sequence_length = getattr(
        tokenizer, "model_max_length", None)
    dataset = load_dataset(dataset_name, split="train")

    def preprocess_function(dataset):
        return tokenizer(dataset[column1], dataset[column2], truncation=True,
                         padding='longest', max_length=max_sequence_length, return_tensors="np")

    tokenized_dataset = dataset.with_transform(preprocess_function)

    log.info("Loaded dataset has %d samples", len(tokenized_dataset))
    return tokenized_dataset


DATASET = InfDataset(_get_dataset(
    DATASET_NAME, DATASET_COLUMN_NAME1, DATASET_COLUMN_NAME2))


class BertUser(TritonUser):
    """Locust User for Bert Dataset"""

    def __init__(self, environment):
        super().__init__(environment, UserContext(
            DATASET, MODEL_NAME, MODEL_VERSION))
