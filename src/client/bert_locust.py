from client.base import DatasetIterator, UserContext
from client.bert import DATASET_NAME, MODEL_NAME, MODEL_VERSION, BertDataset
from client.triton_user import TritonUser


class BertUser(TritonUser):
    """Locust User for Bert Dataset"""

    dataset = DatasetIterator(BertDataset(DATASET_NAME).dataset)

    def __init__(self, environment):
        super().__init__(environment, UserContext(BertUser.dataset, MODEL_NAME, MODEL_VERSION, batch_size=1))
