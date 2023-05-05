from triton_user import FuncUser, Infer
from locust import constant_throughput

class PytorchUser(FuncUser):
     wait_time = constant_throughput(1)  
     def __init__(self, environment):
          super().__init__(environment, "pytorch", Infer().pyt)

