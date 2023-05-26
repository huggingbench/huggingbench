import logging
from client.bert import BertDataset
from client.resnet import ResnetDataset
from client.base import DatasetAlias, DatasetIterator
from client.triton_client import TritonClient

MODEL_DATASET = {'bert-base-uncased': BertDataset,
                 'resnet50': ResnetDataset}

LOG = logging.getLogger(__name__)

def get_dataset( model_name: str) -> DatasetAlias:
    clazz = MODEL_DATASET[model_name]
    instance = clazz()
    return instance.get_dataset()

class RunnerConfig:
    model_name: str
    batch_size: int
    async_req: bool

    def __init__(self, model_name: str, batch_size: int, async_req: bool) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.async_client = async_req

class Runner:

    def __init__(self, cfg: RunnerConfig, client: TritonClient, dataset: DatasetAlias) -> None:
        self.config = cfg
        self.client = client
        self.dataset = DatasetIterator(dataset, infinite=False)
    
    def run(self):
        def send_batch(batch):
            LOG.info("Sending batch of size %d", len(batch))
            if self.config.async_req:
                res = self.client.infer_batch_async(batch)
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
             
         