from base import UserContext, DatasetIterator
from triton_user import TritonUser
from resnet import ResnetDataset, DATASET_NAME, MODEL_NAME, MODEL_VERSION


class ResnetUser(TritonUser):
    # Locust User for Resnet Dataset
    dataset = DatasetIterator(ResnetDataset(DATASET_NAME).dataset)

    def __init__(self, environment):
        super().__init__(environment, UserContext(
            ResnetUser.dataset, MODEL_NAME, MODEL_VERSION))
