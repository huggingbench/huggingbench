import logging
import time
from locust import User, task
from base import UserContext
from triton_client import TritonClient, TRITON_SERVER

LOG = logging.getLogger(__name__)


class TritonUser(User):
    def __init__(self, environment, ctx: UserContext):
        super().__init__(environment)
        self.ctx = ctx
        self.client = TritonClient(TRITON_SERVER, ctx.model_name)

    @task
    def infer(self):
        """Runs inference on the triton server"""
        # Locust event data
        request_meta = {
            "request_type": "infer",
            "name": self.ctx.model_name,
            "start_time": time.time(),
            "response_length": 0,  # calculating this for an xmlrpc.client response would be too hard
            "response": None,
            "context": {},  # see HttpUser if you actually want to implement contexts
            "exception": None,
        }
        try:
            batch = []
            for _ in range(self.ctx.batch_size):
                batch.append(next(self.ctx.dataset))
            start_perf_counter = time.perf_counter()
            LOG.info("Sending batch: %s", batch)
            resp = self.client.infer_batch(batch)
        except Exception as err:
            LOG.error("Exception: %s", err, exec_info=True)
            request_meta["exception"] = err

        request_meta["response_time"] = (time.perf_counter() - start_perf_counter) * 1000
        request_meta["response"] = resp.get_response()
        self.environment.events.request.fire(**request_meta)
