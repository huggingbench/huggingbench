import logging
import json
from typing import List
from prometheus_client import REGISTRY, start_http_server, Counter, Histogram, Info, write_to_textfile
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
import numpy as np

### For dev purposes log INFO level
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__file__)
TRITON_SERVER = "localhost:8000"
MODEL_VERSION = "1"
PROM_PORT = 8011  # Prometheus port


class TritonClient:
    """ Runs the given function in locust and records metrics """
    metric_infer_latency = Histogram(
        "infer_latency", "Latency for inference in seconds", labelnames=["model", "batch_size"])
    metric_infer_requests = Counter(
        "infer_requests", "Number of inference requests", labelnames=["model", "batch_size"])
    metric_info = Info("client_info", "Information about the client")
    prom_started = False

    def __init__(self, triton_url: str, model_name: str, max_paralell_requests: int = 1, prom_port: int = PROM_PORT):
        self._server_url = triton_url
        # we can't have "/" in the model file path
        self.model = model_name
        self.model_version = MODEL_VERSION
        self.metric_info.info({"model": self.model})
        LOG.info("Creating triton client for server: %s", self._server_url)
        self.client = httpclient.InferenceServerClient(
            url=self._server_url, concurrency=max_paralell_requests)
        errors = self._server_check(self.client)
        if errors:
            raise RuntimeError(f"Triton Server check failed: {errors}")
        LOG.info("Triton server check successful. Server is ready to handle requests")
        model_config = self.client.get_model_config(
            self.model, self.model_version)
        model_metadata = self.client.get_model_metadata(
            self.model, self.model_version)
        LOG.info("Model config: %s", json.dumps(model_config, indent=4))
        LOG.info("Model metadata: %s", json.dumps(model_metadata, indent=4))

        self.inputs = {tm["name"]: tm for tm in model_metadata["inputs"]}
        self.outputs = {tm["name"]: tm for tm in model_metadata["outputs"]}

        if not TritonClient.prom_started:
            LOG.info("Starting Prometheus server on port %s", prom_port)
            start_http_server(prom_port)
            TritonClient.prom_started = True

    def infer(self, sample) -> httpclient.InferResult:
        """ Runs inference on the triton server """
        batched_sample = [sample]
        return self.infer_batch(batched_sample)
    
    def infer_batch(self, samples) -> httpclient.InferResult:
        return self._infer_batch(samples, async_req=False)

    def _infer_batch(self, samples, async_req : bool = False):
        """ Runs inference on the triton server """
        self.metric_infer_requests.labels(self.model, len(samples)).inc()
        infer_inputs = []
        infer_outputs = []
        batched_data_per_input = {}
        for sample in samples:
            for input in self.inputs:
                data_type = self.inputs[input]["datatype"]
                np_dtype = triton_to_np_dtype(
                    data_type)
                data = sample[input].astype(np_dtype)
                if input not in batched_data_per_input:
                    batched_data_per_input[input] = []
                batched_data_per_input[input].append(data)
                LOG.debug("Input: %s, shape: %s", input, data.shape)
        stacked_batched_data_per_input = {}
        for input in batched_data_per_input:
            stacked_batched_data_per_input[input] = np.stack(
                batched_data_per_input[input], 0)
        for stacked_input in stacked_batched_data_per_input:
            infer_input = httpclient.InferInput(stacked_input,
                                                stacked_batched_data_per_input[stacked_input].shape, self.inputs[stacked_input]["datatype"])
            infer_input.set_data_from_numpy(
                stacked_batched_data_per_input[stacked_input])
            infer_inputs.append(infer_input)
        infer_outputs = self._prepare_infer_outputs()
        with self.metric_infer_latency.labels(self.model, len(samples)).time():
            if async_req:
                return self.client.async_infer(
                model_name=self.model, model_version=self.model_version,
                inputs=infer_inputs, outputs=infer_outputs)
            else:
                return self.client.infer(
                    model_name=self.model, model_version=self.model_version,
                inputs=infer_inputs, outputs=infer_outputs)
    
    def infer_batch_async(self, samples) -> httpclient.InferAsyncRequest:
        return self._infer_batch(samples, async_req=True)


    def _prepare_infer_inputs(self, sample) -> List[httpclient.InferInput]:
        """ Prepares the input for inference """
        infer_inputs = []
        for input in self.inputs:
            data_type = self.inputs[input]["datatype"]
            np_dtype = triton_to_np_dtype(
                data_type)
            data = sample[input].astype(np_dtype)
            # adding batch dimension
            data = np.expand_dims(data, axis=0)
            infer_input = httpclient.InferInput(
                input, data.shape, data_type)
            infer_input.set_data_from_numpy(
                data)
            infer_inputs.append(infer_input)
        return infer_inputs

    def _prepare_infer_outputs(self) -> List[httpclient.InferRequestedOutput]:
        """ Prepares the output for inference """
        infer_outputs = []
        for output in self.outputs:
            infer_outputs.append(
                httpclient.InferRequestedOutput(output))
        return infer_outputs

    def infer_req(self, sample) -> httpclient.InferResult:
        infer_inputs = self._prepare_infer_inputs(sample)
        infer_outputs = self._prepare_infer_outputs()
        with self.metric_infer_latency.labels(self.model).time():
            infer_res: httpclient.InferResult = self.client.infer(
                model_name=self.model, model_version=self.model_version,
                inputs=infer_inputs, outputs=infer_outputs)
            return infer_res

    def infer_req_async(self, sample) -> httpclient.InferAsyncRequest:
        infer_inputs = self._prepare_infer_inputs(sample)
        infer_outputs = self._prepare_infer_outputs(sample)
        return self.client.async_infer(model_name=self.model, model_version=self.model_version,
                                       inputs=infer_inputs, outputs=infer_outputs)

    def process_async_requests(self, async_requests: List[httpclient.InferAsyncRequest]) -> List[httpclient.InferResult]:
        """ Processes the async requests and returns the results """
        results = []
        for async_req in async_requests:
            results.append(async_req.get_result())
        return results

    def _server_check(self, client):
        errors = []
        if not client.is_server_live():
            errors.append(f"Triton server {self._server_url} is not up")
        if not client.is_server_ready():
            errors.append(f"Triton server {self._server_url} is not ready")
        if not client.is_model_ready(self.model, self.model_version):
            errors.append(
                f"Model {self.model}:{self.model_version} is not ready")
        return errors
    
    def write_metrics(file: str):
        write_to_textfile(file, REGISTRY)
