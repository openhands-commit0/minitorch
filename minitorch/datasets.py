import math
import random
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]
datasets = {'Simple': simple, 'Diag': diag, 'Split': split, 'Xor': xor, 'Circle': circle, 'Spiral': spiral}