from client.base import DatasetAlias
from client.bert import BertDataset, BertGenDataset, DistilBertGenDataset
from client.resnet import ResnetDataset, ResnetGenDataset

MODEL_DATASET = {
    "bert-base-uncased": BertDataset,
    "microsoft/resnet-50": ResnetDataset,
    "bert-base-uncased-gen": BertGenDataset,
    "microsoft/resnet-50-gen": ResnetGenDataset,
    "distilbert-base-uncased-gen": DistilBertGenDataset,
}


def get_dataset(name: str) -> DatasetAlias:
    clazz = MODEL_DATASET[name]
    if clazz is None:
        raise ValueError("No dataset found for '%s'", name)
    instance = clazz()
    return instance.get_dataset()
