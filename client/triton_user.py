import logging
import time
from locust import User, task, between
from base import UserContext
from triton_client import TritonClient, TRITON_SERVER

LOG = logging.getLogger(__name__)


class TritonUser(User):

    wait_time = between(1, 5)

    def __init__(self, environment, ctx: UserContext):
        super().__init__(environment)
        self.model = ctx.model_name
        self.client = TritonClient(TRITON_SERVER, ctx.model_name)
        self.dataset = ctx.dataset

    @task
    def infer(self):
        """ Runs inference on the triton server """
        # Locust event data
        request_meta = {
            "request_type": "infer",
            "name": self.model,
            "start_time": time.time(),
            "response_length": 0,  # calculating this for an xmlrpc.client response would be too hard
            "response": None,
            "context": {},  # see HttpUser if you actually want to implement contexts
            "exception": None,
        }
        try:
            sample = next(self.dataset)
            start_perf_counter = time.perf_counter()
            resp = self.client.infer(sample)
        except Exception as err:
            LOG.error("Exception: %s", err, exec_info=True)
            request_meta["exception"] = err

        request_meta["response_time"] = (
            time.perf_counter() - start_perf_counter) * 1000
        request_meta["response"] = resp.get_response()
        self.environment.events.request.fire(**request_meta)
