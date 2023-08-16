from dataclasses import dataclass
import logging
import os
import queue
from concurrent.futures import (
    CancelledError,
    Future,
    ThreadPoolExecutor,
    TimeoutError,
)
import concurrent.futures as concurrent_futures
from threading import Event, Lock, Thread
from timeit import default_timer as timer
from typing import List

from tritonclient.http import InferenceServerException, InferAsyncRequest, InferResult
from bench.plugin import Client

from client.base import DatasetAlias, DatasetIterator

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
    total: int  # total number of inferences
    success_count: int  # number of successful inferences

    def __str__(self) -> str:
        return f"RunnerStats(execution_times={self.execution_times}, success_rate={self.success_rate}, failure_rate={self.failure_rate}, total={self.total}, success_count={self.success_count})"

    def merge(stats):
        """Merge multiple RunnerStats into a single one"""
        execution_times = stats[0].execution_times
        success_rate = 0
        failure_rate = 0
        total = 0
        success_count = 0
        for s in stats:
            execution_times.extend(s.execution_times)
            success_rate += s.success_rate
            failure_rate += s.failure_rate
            total += s.total
            success_count += s.success_count
        return RunnerStats(execution_times, success_rate, failure_rate, total, success_count)


ENV_EXPERIMENT_RUN_INTERVAL = "EXPERIMENT_RUN_INTERVAL"
EXPERIMENT_RUN_INTERVAL = 150  # for how long to run the experiment in seconds
LOG_PROGRESS_MSG_INTERVAL = 10  # log progress info every 10 seconds


class Runner:
    """Runner is responsible for sending requests to the server using
    inference client and collecting the results.
    It manages client concurrency and async behavior."""

    def __init__(self, idx: int, cfg: RunnerConfig, client: Client, dataset: DatasetAlias, time_bound=True) -> None:
        self.idx = idx
        self.config = cfg
        self.client = client
        self.dataset = DatasetIterator(dataset, infinite=time_bound)
        self.lock = Lock()
        self.execution_times = []
        self.experiment_run_interval = int(os.getenv(ENV_EXPERIMENT_RUN_INTERVAL, EXPERIMENT_RUN_INTERVAL))

    def run(self) -> RunnerStats:
        LOG.info(f"Starting client runner {self.idx}")
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
                req = self.client.infer(batch, async_req=True)
                if req is not None:
                    async_reqs.put(req)
                    LOG.debug("Sent async batch request")
                    success = True
                else:
                    LOG.warn("Failed async batch request")
            else:
                res = self.client.infer(batch)
                if res is not None:
                    LOG.debug("Received batch response")
                    success = True
                else:
                    LOG.info("Failed batch request")
            end = timer()
            self.lock.acquire()
            self.execution_times.append(end - start)  # this is only true for sync requests
            self.lock.release()
            return success

        if self.config.async_req:

            def get_async_result(async_reqs: queue.Queue[InferAsyncRequest], completed: Event, batch_size: int):
                while not completed.is_set():
                    req = async_reqs.get()
                    try:
                        res: InferResult = req.get_result()
                        LOG.debug("Received async result: %s", res.get_response())
                        success_counter.increment(
                            batch_size
                        )  #  this is not fully accurate b/c the last batch might not be full, the proper workaround is to inspect the response
                    except InferenceServerException as e:
                        LOG.warn("Failed async request: %s", e.debug_details())
                        fail_counter.increment(batch_size)

            async_res_thread = Thread(
                target=get_async_result, args=(async_reqs, completed, self.config.batch_size)
            )  # process async responses
            async_res_thread.daemon = True  # queue.get() is blocking, so we need to make sure this thread is killed
            async_res_thread.start()

        total = 0

        def future_result(f: Future, batch_size: int):
            try:
                success = f.result()
                LOG.debug("Future completed with result: %s", success)
                if not self.config.async_req:
                    """If not async, then we need to increment the counters here"""
                    success_counter.increment(batch_size) if success else fail_counter.increment(batch_size)
            except (CancelledError, TimeoutError, Exception) as e:
                LOG.error("future error: %s", e)
                fail_counter.increment(batch_size)

        batch = []
        first_request_time = timer()
        log_info_update_time = first_request_time + LOG_PROGRESS_MSG_INTERVAL  # log info every 15 seconds
        max_time = first_request_time + self.experiment_run_interval
        LOG.info("Client %d: Running experiment for %d seconds", self.idx, self.experiment_run_interval)

        batches = []  # we process dataset in chunk of batches
        for sample in self.dataset:
            cur_time = timer()
            if cur_time > max_time:
                LOG.info(
                    "Client %d: Stopping experiment benchmark. Reached experiment time limit of %d seconds",
                    self.idx,
                    self.experiment_run_interval,
                )
                LOG.info("Client %d: processed total of %d items", self.idx, total)
                break
            if cur_time > log_info_update_time:
                LOG.info(
                    "Client %d: processed %d items in %d seconds. Success rate: %f, Failure rate: %f ...",
                    self.idx,
                    total,
                    cur_time - first_request_time,
                    success_counter.value() / (cur_time - first_request_time),
                    fail_counter.value() / (cur_time - first_request_time),
                )
                log_info_update_time = cur_time + LOG_PROGRESS_MSG_INTERVAL
                if fail_counter.value() > 0:
                    if success_counter.value() / (success_counter.value() + fail_counter.value()) < 0.05:
                        LOG.warn("Success rate is less than 5%. Stopping the experiment")
                        break
            total += 1
            batch.append(sample)
            if len(batch) == self.config.batch_size:
                batches.append(batch)
                batch = []
            if len(batches) >= 100:  # arbitrary number of batches to process in parallel
                futures = [executor.submit(send_batch, b) for b in batches]
                for f in concurrent_futures.as_completed(futures):
                    future_result(f, self.config.batch_size)

        if len(batches) > 0:
            futures = [executor.submit(send_batch, b) for b in batches]
            for f in concurrent_futures.as_completed(futures):
                future_result(f, self.config.batch_size)

        if len(batch) > 0:
            futures = [executor.submit(send_batch, batch)]
            for f in concurrent_futures.as_completed(futures):
                future_result(f, len(batch))

        LOG.info(f"Client {self.idx}: Processed all items")
        executor.shutdown(wait=True, cancel_futures=True)
        if fail_counter.value() > 0:
            LOG.warn("Failed %d requests", fail_counter.value())
        completed.set()
        last_request_time = timer()
        success_rate = success_counter.value() / (last_request_time - first_request_time)
        failure_rate = fail_counter.value() / (last_request_time - first_request_time)
        if success_rate == 0:
            LOG.error(f"Client {self.idx}: No successful inference requests")
            return None
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
