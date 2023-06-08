import logging
import queue
from threading import Thread, Event, Lock
from concurrent.futures import Future, ThreadPoolExecutor
from client.base import DatasetAlias, DatasetIterator
from client.triton_client import TritonClient
from timeit import default_timer as timer
from tritonclient.http import InferenceServerException

LOG = logging.getLogger(__name__)

class RunnerConfig:

    def __init__(self, batch_size: int = 1, async_req: bool = False, workers: int = 1) -> None:
        self.batch_size = batch_size
        self.async_req = async_req
        self.workers = workers

class Runner:

    def __init__(self, cfg: RunnerConfig, client: TritonClient, dataset: DatasetAlias) -> None:
        self.config = cfg
        self.client = client
        self.dataset = DatasetIterator(dataset, infinite=False)
        self.execution_times = []
    
    def run(self):
        LOG.info("Starting client runner")
        async_reqs = queue.Queue(maxsize=200) # Size picked arbitrarily. Sets limit on number of outstanding requests
        completed = Event()
        executor = ThreadPoolExecutor(max_workers=self.config.workers)
        fail_counter = ThreadSafeCounter()
        success_counter = ThreadSafeCounter()
        def send_batch(batch):
            LOG.debug("Sending batch of size %d", len(batch))
            start = timer()
            success = False
            if self.config.async_req:
                req = self.client.infer_batch_async(batch)
                if req is not None:
                    async_reqs.put(req)
                    LOG.debug("Sent async request")
                    success = True
                else:
                    LOG.warn("Failed async request")
            else:
                res = self.client.infer_batch(batch)
                if res is not None:
                    LOG.debug("Received response")
                    success = True
                else:
                    LOG.warn("Failed request")
            end = timer()
            self.execution_times.append(end - start)
            batch.clear()
            return success
        
        if self.config.async_req:
            def get_async_result(async_reqs: queue.Queue, completed: Event):
                while not completed.is_set():
                    req = async_reqs.get()
                    try:
                        res = req.get_result()
                        LOG.debug("Received async result: %s", res.get_response())
                        success_counter.increment(1)
                    except InferenceServerException as e:
                        LOG.warn("Failed async request: %s", e.debug_details())
                        fail_counter.increment(1)
                    
            Thread(target=get_async_result, args=(async_reqs, completed)).start() # process async responses

        def future_status(f: Future):
            success = f.result()
            success_counter.increment(1) if success else fail_counter.increment(1)
            
        batch = []
        total = len(self.dataset)
        LOG.info("Processed 0 of %d items", total)
        item_cnt = 0
        shard = 1
        for sample in self.dataset:
            item_cnt+=1
            progress = item_cnt/total
            if progress > 0.1:
                LOG.info(f"Processed {int(progress*shard*100)}%...", )
                item_cnt = 0
                shard+=1

            batch.append(sample)
            if len(batch) == self.config.batch_size:
                f = executor.submit(send_batch, batch)
                f.add_done_callback(future_status)

        if len(batch) > 0:
            f = executor.submit(send_batch, batch)
            f.add_done_callback(future_status)
        LOG.info("Processed all items")
        LOG.info("Finished client runner")
        executor.shutdown(wait=True, cancel_futures=False)
        if fail_counter.value() > 0:
            LOG.warn("Failed %d requests", fail_counter.value())
        completed.set()
          # Convert execution times to a numpy array
        execution_times = self.execution_times
        self.execution_times = []
        return execution_times
             

class ThreadSafeCounter():

    def __init__(self):
        self._counter = 0
        self._lock = Lock()
 
    def increment(self, val):
        with self._lock:
            self._counter += val
 
    def value(self):
        with self._lock:
            return self._counter
         