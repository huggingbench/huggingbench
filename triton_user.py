# from triton_client import *
import time
from locust import User, task
from prometheus_client import start_http_server, Counter, Histogram, Info
from torchvision import transforms
from PIL import Image
import tritonclient.http as httpclient
import numpy as np
from tritonclient.utils import triton_to_np_dtype
import os

# Runs the given function in locust and records metrics
class FuncUser(User):
    # wait_time = constant_pacing(2)  # call the task every 2 seconds
    abstract = True
    metric_infer_latency = Histogram("infer_latency", "Latency for inference in seconds", labelnames=["model"])
    metric_infer_requests = Counter("infer_requests", "Number of inference requests", labelnames= ["model"])
    metric_info = Info("client_info", "Information about the client")
    print("Starting Prometheus server on port 8001")
    start_http_server(8001)
    
    def __init__(self, environment, model, func):
         super().__init__(environment)
         self.infer = Infer()
         self.model = model
         self.metric_info.info({"model": model})
         self.func = func
    
    @task
    def infer(self):
        self.metric_infer_requests.labels(self.model).inc()
        start_time = time.time()
        with self.metric_infer_latency.labels(self.model).time():
            # Locust event data
            request_meta = {
                "request_type": "infere",
                "name": self.model,
                "start_time": time.time(),
                "response_length": 0,  # calculating this for an xmlrpc.client response would be too hard
                "response": None,
                "context": {},  # see HttpUser if you actually want to implement contexts
                "exception": None,
            }
            start_perf_counter = time.perf_counter()
            try:
                infer_res : httpclient.InferResult = self.func()
                request_meta["response"] = infer_res.get_response()
            except Exception as e:
                request_meta["exception"] = e
                
            request_meta["response_time"] = (time.perf_counter() - start_perf_counter) * 1000
            self.environment.events.request.fire(**request_meta)

# Runs inference on the https://huggingface.co/microsoft/resnet-50 model
class Infer:
    
    def __init__(self) -> None:
        self.rn5_input = self.rn50_preprocess()
        self.triton_client = self.create_client()

    def rn50_preprocess(self, img_path="img1.jpg"):
        img = Image.open(img_path)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(img).numpy()

    def create_client(self):
        return httpclient.InferenceServerClient(url="localhost:8000")

    def infer(self, triton_client: httpclient.InferenceServerClient , input, mn):
        # Setup a connection with the Triton Inference Server. 
        # Specify the names of the input and output layer(s) of our model.
        infer_input = httpclient.InferInput("pixel_values", [1,3, 224, 224], datatype="FP32")
        input = np.expand_dims(input, axis=0) # adding batch dimension
        infer_input.set_data_from_numpy(input, binary_data=True)
        test_output = httpclient.InferRequestedOutput("logits", binary_data=True, class_count=1000)

        # Querying the server
        infer_res = triton_client.infer(model_name=mn, inputs=[infer_input], outputs=[test_output])
        if infer_res.as_numpy('logits').size == 0:
            raise ValueError("Received unexpected response from the server.")
        # test_output_fin = infer_res.as_numpy('logits')
        # return test_output_fin[:5]
        return infer_res
    
    def pyt(self):
        return self.infer(self.triton_client, self.rn5_input, "resnet50_torch")

    def onnx(self):
        return self.infer(self.triton_client, self.rn5_input, "resnet50-onnx")
