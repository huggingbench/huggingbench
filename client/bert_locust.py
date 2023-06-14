from client.triton_user import TritonUser
from client.base import UserContext, DatasetIterator
from client.bert import BertDataset, DATASET_NAME, MODEL_NAME, MODEL_VERSION


class BertUser(TritonUser):
    """Locust User for Bert Dataset"""

    dataset = DatasetIterator(BertDataset(DATASET_NAME).dataset)

    def __init__(self, environment):
        super().__init__(environment, UserContext(BertUser.dataset, MODEL_NAME, MODEL_VERSION, batch_size=1))
