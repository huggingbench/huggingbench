# from triton_client import *
from locust import User, task
import time
import locust.stats
from torchvision import transforms
from PIL import Image
import tritonclient.http as httpclient
import numpy as np
from tritonclient.utils import triton_to_np_dtype

locust.stats.CSV_STATS_INTERVAL_SEC = 5 # default is 1 second
locust.stats.CSV_STATS_FLUSH_INTERVAL_SEC = 10 # Determines how often the data is flushed to disk, default is 10 seconds
     
# Runs the given function in loccust and records metrics
class FuncUser(User):
    # wait_time = constant_pacing(2)  # call the task every 2 seconds
    abstract = True
    def __init__(self, environment, name, func):
         super().__init__(environment)
         self.func = func
         self.name = name
    
    @task
    def my_task(self):
        start_time = time.monotonic()  # record the start time
        result = self.func()  # call your custom method here
        end_time = time.monotonic()  # record the end time
        latency = (end_time - start_time) * 1000  # calculate the latency in milliseconds
        self.log_event(name=self.name, response_time=latency, response_length=len(result))  # log the latency

    def log_event(self, name, response_time, response_length):
        event_data = {
            "request_type": "request",
            "name": name,
            "response_time": response_time,
            "response_length": response_length,
            
        }
        self.environment.events.request.fire(**event_data)


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

    def infer(self, triton_client, input, mn):
        # Setup a connection with the Triton Inference Server. 
        # Specify the names of the input and output layer(s) of our model.
        test_input = httpclient.InferInput("input", input.shape, datatype="FP32")
        test_input.set_data_from_numpy(input, binary_data=True)
        test_output = httpclient.InferRequestedOutput("output", binary_data=True, class_count=1000)

        # Querying the server
        results = triton_client.infer(model_name=mn, inputs=[test_input], outputs=[test_output])
        test_output_fin = results.as_numpy('output')
        return test_output_fin[:5]
    
    def pyt(self):
        return self.infer(self.triton_client, self.rn5_input, "resnet50_torch")

    def onnx(self):
        return self.infer(self.triton_client, self.rn5_input, "resnet50_onnx")
    