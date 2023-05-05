from triton_user import FuncUser, Infer
from locust import constant_throughput
import time

class OnnxUser(FuncUser):
     wait_time = constant_throughput(1)     
     def __init__(self, environment):
          super().__init__(environment, "onnx", Infer().onnx)

