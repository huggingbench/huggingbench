from triton_user import FuncUser, Infer

class OnnxUser(FuncUser):
     abstract = False 
     def __init__(self, environment):
          super().__init__(environment, "onnx", Infer().onnx)

