import logging
from client.bert import BertDataset, BertGenDataset, DistilBertGenDataset
from client.resnet import ResnetDataset, ResnetGenDataset
from client.base import DatasetAlias, DatasetIterator
from client.triton_client import TritonClient

MODEL_DATASET = {'bert-base-uncased': BertDataset,
                 'microsoft/resnet-50': ResnetDataset,
                 'bert-base-uncased-gen': BertGenDataset,
                 'microsoft/resnet-50-gen': ResnetGenDataset,
                 'distilbert-base-uncased-gen': DistilBertGenDataset}

LOG = logging.getLogger(__name__)

def get_dataset(name: str) -> DatasetAlias:
    clazz = MODEL_DATASET[name]
    if clazz is None:
        raise ValueError("No dataset found for '%s'", name)
    instance = clazz()
    return instance.get_dataset()

class RunnerConfig:

    def __init__(self, batch_size: int = 1, async_req: bool = False) -> None:
        self.batch_size = batch_size
        self.async_req = async_req

class Runner:

    def __init__(self, cfg: RunnerConfig, client: TritonClient, dataset: DatasetAlias) -> None:
        self.config = cfg
        self.client = client
        self.dataset = DatasetIterator(dataset, infinite=False)
    
    def run(self):
        LOG.info("Starting client runner")
        def send_batch(batch):
            LOG.debug("Sending batch of size %d", len(batch))
            if self.config.async_req:
                res = self.client.infer_batch_async(batch)
                LOG.debug("Received response %s", res.get_result())
            else:
                res = self.client.infer_batch(batch)
                LOG.debug("Received response %s", res.get_response())
            batch.clear()
        
        batch = []
        for sample in self.dataset:
            batch.append(sample)
            if len(batch) == self.config.batch_size:
              send_batch(batch)
        if len(batch) > 0:
            send_batch(batch)
        LOG.info("Finished client runner")
             
         