"""
Collection of the core mathematical operators used throughout the code base.
"""
import math
from typing import Callable, Iterable

def mul(x: float, y: float) -> float:
    """$f(x, y) = x * y$"""
    pass

def id(x: float) -> float:
    """$f(x) = x$"""
    pass

def add(x: float, y: float) -> float:
    """$f(x, y) = x + y$"""
    pass

def neg(x: float) -> float:
    """$f(x) = -x$"""
    pass

def lt(x: float, y: float) -> float:
    """$f(x) =$ 1.0 if x is less than y else 0.0"""
    pass

def eq(x: float, y: float) -> float:
    """$f(x) =$ 1.0 if x is equal to y else 0.0"""
    pass

def max(x: float, y: float) -> float:
    """$f(x) =$ x if x is greater than y else y"""
    pass

def is_close(x: float, y: float) -> float:
    """$f(x) = |x - y| < 1e-2$"""
    pass

def sigmoid(x: float) -> float:
    """
    $f(x) =  \\frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \\frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\\frac{e^x}{(1.0 + e^{x})}$

    for stability.
    """
    pass

def relu(x: float) -> float:
    """
    $f(x) =$ x if x is greater than 0, else 0

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)
    """
    pass
EPS = 1e-06

def log(x: float) -> float:
    """$f(x) = log(x)$"""
    pass

def exp(x: float) -> float:
    """$f(x) = e^{x}$"""
    pass

def log_back(x: float, d: float) -> float:
    """If $f = log$ as above, compute $d \\times f'(x)$"""
    pass

def inv(x: float) -> float:
    """$f(x) = 1/x$"""
    pass

def inv_back(x: float, d: float) -> float:
    """If $f(x) = 1/x$ compute $d \\times f'(x)$"""
    pass

def relu_back(x: float, d: float) -> float:
    """If $f = relu$ compute $d \\times f'(x)$"""
    pass

def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
         A function that takes a list, applies `fn` to each element, and returns a
         new list
    """
    pass

def negList(ls: Iterable[float]) -> Iterable[float]:
    """Use `map` and `neg` to negate each element in `ls`"""
    pass

def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
         Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """
    pass

def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add the elements of `ls1` and `ls2` using `zipWith` and `add`"""
    pass

def reduce(fn: Callable[[float, float], float], start: float) -> Callable[[Iterable[float]], float]:
    """
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
         Function that takes a list `ls` of elements
         $x_1 \\ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """
    pass

def sum(ls: Iterable[float]) -> float:
    """Sum up a list using `reduce` and `add`."""
    pass

def prod(ls: Iterable[float]) -> float:
    """Product of a list using `reduce` and `mul`."""
    pass