"""
Implementation of the autodifferentiation Functions for Tensor.
"""
from __future__ import annotations
import random
from typing import TYPE_CHECKING
import numpy as np
import minitorch
from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend
if TYPE_CHECKING:
    from typing import Any, List, Tuple
    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape

def wrap_tuple(x):
    """Turn a possible value into a tuple"""
    pass

class Function:
    pass

class Neg(Function):
    pass

class Inv(Function):
    pass

class Add(Function):
    pass

class Mul(Function):
    pass

class Sigmoid(Function):
    pass

class ReLU(Function):
    pass

class Log(Function):
    pass

class Exp(Function):
    pass

class Sum(Function):
    pass

class All(Function):
    pass

class LT(Function):
    pass

class EQ(Function):
    pass

class IsClose(Function):
    pass

class Permute(Function):
    pass

class View(Function):
    pass

class Copy(Function):
    pass

class MatMul(Function):
    pass

def zeros(shape: UserShape, backend: TensorBackend=SimpleBackend) -> Tensor:
    """
    Produce a zero tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend

    Returns:
        new tensor
    """
    pass

def rand(shape: UserShape, backend: TensorBackend=SimpleBackend, requires_grad: bool=False) -> Tensor:
    """
    Produce a random tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    pass

def _tensor(ls: Any, shape: UserShape, backend: TensorBackend=SimpleBackend, requires_grad: bool=False) -> Tensor:
    """
    Produce a tensor with data ls and shape `shape`.

    Args:
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
        new tensor
    """
    pass

def tensor(ls: Any, backend: TensorBackend=SimpleBackend, requires_grad: bool=False) -> Tensor:
    """
    Produce a tensor with data and shape from ls

    Args:
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    pass