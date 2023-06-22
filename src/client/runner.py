from dataclasses import dataclass
import logging
import queue
from concurrent.futures import (
    CancelledError,
    Future,
    ThreadPoolExecutor,
    TimeoutError,
    as_completed,
)
from threading import Event, Lock, Thread
from timeit import default_timer as timer
from typing import List

from tritonclient.http import InferenceServerException

from client.base import DatasetAlias, DatasetIterator
from client.triton_client import TritonClient

LOG = logging.getLogger(__name__)


class RunnerConfig:
    def __init__(self, batch_size: int = 1, async_req: bool = False, workers: int = 1) -> None:
        self.batch_size = batch_size
        self.async_req = async_req
        self.workers = workers

@dataclass
class RunnerStats:
    execution_times: List[float]
    success_rate: float
    failure_rate: float
    total: int
    success_count: int

    def __str__(self) -> str:
        return f"RunnerStats(execution_times={self.execution_times}, success_rate={self.success_rate}, failure_rate={self.failure_rate}, total={self.total}, success_count={self.success_count})"


class Runner:
    """Runner is responsible for sending requests to the server using
    inference client and collecting the results.
    It manages client concurrency and async behavior."""

    def __init__(self, cfg: RunnerConfig, client: TritonClient, dataset: DatasetAlias) -> None:
        self.config = cfg
        self.client = client
        self.dataset = DatasetIterator(dataset, infinite=False)
        self.execution_times = []

    def run(self) -> RunnerStats:
        LOG.info("Starting client runner")
        async_reqs = queue.Queue(maxsize=1000)  # Size picked arbitrarily. Sets limit on number of outstanding requests
        completed = Event()
        executor = ThreadPoolExecutor(max_workers=self.config.workers)
        fail_counter = ThreadSafeCounter()
        success_counter = ThreadSafeCounter()

        def send_batch(batch):
            if len(batch) == 0:
                LOG.warn("Attempted sending batch with no data")
                return

            LOG.debug("Sending batch of size %d", len(batch))
            start = timer()
            success = False
            if self.config.async_req:
                req = self.client.infer_batch_async(batch)
                if req is not None:
                    async_reqs.put(req)
                    LOG.debug("Sent async batch request")
                    success = True
                else:
                    LOG.warn("Failed async batch request")
            else:
                res = self.client.infer_batch(batch)
                if res is not None:
                    LOG.debug("Received batch response")
                    success = True
                else:
                    LOG.info("Failed batch request")
            end = timer()
            self.execution_times.append(end - start)  # this is only true for sync requests
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

            async_res_thread = Thread(target=get_async_result, args=(async_reqs, completed))  # process async responses
            async_res_thread.daemon = True  # queue.get() is blocking, so we need to make sure this thread is killed
            async_res_thread.start()

        item_cnt = 0
        batch_group_cnt = 0
        total = len(self.dataset)
        status_update_lock = Lock()

        def future_result(f: Future):
            nonlocal item_cnt, batch_group_cnt, total
            try:
                success = f.result()
                LOG.debug("Future completed with result: %s", success)
                if not self.config.async_req:
                    """If not async, then we need to increment the counters here"""
                    success_counter.increment(1) if success else fail_counter.increment(1)
            except (CancelledError, TimeoutError, Exception) as e:
                LOG.error("future error: %s", e)
                fail_counter.increment(1)
            finally:
                with status_update_lock:
                    item_cnt += 1
                    progress = item_cnt / total
                    if progress > 0.1:
                        LOG.info(
                            f"Processed {int(progress*batch_group_cnt*100)}%...",
                        )
                        item_cnt = 0
                        batch_group_cnt += 1

        batch = []
        futures = []
        first_request_time = timer()
        LOG.info("Processed 0 of %d items", total)
        for sample in self.dataset:
            batch.append(sample)
            if len(batch) == self.config.batch_size:
                f = executor.submit(send_batch, batch)
                futures.append(f)
                batch = []

        if len(batch) > 0:
            f = executor.submit(send_batch, batch)
            futures.append(f)

        for f in as_completed(futures):
            future_result(f)
        LOG.info("Processed all items")
        executor.shutdown(wait=True, cancel_futures=True)
        LOG.info("Finished processing all items")
        if fail_counter.value() > 0:
            LOG.warn("Failed %d requests", fail_counter.value())
        completed.set()
        last_request_time = timer()
        success_rate = success_counter.value() / (last_request_time - first_request_time)
        failure_rate = fail_counter.value() / (last_request_time - first_request_time)
        # Convert execution times to a numpy array
        execution_times = self.execution_times
        self.execution_times = []
        return RunnerStats(execution_times, success_rate, failure_rate, total, success_counter.value())


class ThreadSafeCounter:
    def __init__(self, val=0):
        self._counter = val
        self._lock = Lock()

    def increment(self, val):
        with self._lock:
            self._counter += val

    def value(self):
        with self._lock:
            return self._counter

    def set(self, val):
        with self._lock:
            self._counter = val
