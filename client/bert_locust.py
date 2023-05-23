from triton_user import TritonUser
from base import UserContext, InfDataset
from bert import BertDataset, DATASET_NAME, MODEL_NAME, MODEL_VERSION


class BertUser(TritonUser):
    """Locust User for Bert Dataset"""
    dataset = InfDataset(BertDataset(DATASET_NAME).dataset)

    def __init__(self, environment):
        super().__init__(environment, UserContext(
            BertUser.dataset, MODEL_NAME, MODEL_VERSION))
