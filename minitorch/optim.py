from typing import Sequence
from .module import Parameter
from .scalar import Scalar

class Optimizer:

    def __init__(self, parameters: Sequence[Parameter]):
        self.parameters = parameters

class SGD(Optimizer):

    def __init__(self, parameters: Sequence[Parameter], lr: float=1.0):
        super().__init__(parameters)
        self.lr = lr