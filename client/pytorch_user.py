from triton_user import FuncUser, Infer

class PytorchUser(FuncUser):
     abstract = False
     def __init__(self, environment):
          super().__init__(environment, "pytorch", Infer().pyt)

