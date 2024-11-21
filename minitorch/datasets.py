import math
import random
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]

def simple(N: int = 100) -> Graph:
    """Simple dataset with two linearly separable clouds of points"""
    X = []
    y = []
    for i in range(N):
        x = random.uniform(-1.0, 1.0)
        y_val = random.uniform(-1.0, 1.0)
        label = 1 if x + y_val > 0 else 0
        X.append((x, y_val))
        y.append(label)
    return Graph(N, X, y)

def diag(N: int = 100) -> Graph:
    """Dataset with two diagonal lines of points"""
    X = []
    y = []
    for i in range(N):
        if random.random() > 0.5:
            x = random.uniform(-1.0, 1.0)
            y_val = x + 0.2 * random.uniform(-1.0, 1.0)
            label = 1
        else:
            x = random.uniform(-1.0, 1.0)
            y_val = -x + 0.2 * random.uniform(-1.0, 1.0)
            label = 0
        X.append((x, y_val))
        y.append(label)
    return Graph(N, X, y)

def split(N: int = 100) -> Graph:
    """Dataset with two distinct regions"""
    X = []
    y = []
    for i in range(N):
        x = random.uniform(-1.0, 1.0)
        y_val = random.uniform(-1.0, 1.0)
        if x < 0:
            label = 1 if y_val > 0 else 0
        else:
            label = 1 if y_val < 0 else 0
        X.append((x, y_val))
        y.append(label)
    return Graph(N, X, y)

def xor(N: int = 100) -> Graph:
    """Dataset with XOR pattern"""
    X = []
    y = []
    for i in range(N):
        x = random.uniform(-1.0, 1.0)
        y_val = random.uniform(-1.0, 1.0)
        if (x > 0 and y_val > 0) or (x < 0 and y_val < 0):
            label = 1
        else:
            label = 0
        X.append((x, y_val))
        y.append(label)
    return Graph(N, X, y)

def circle(N: int = 100) -> Graph:
    """Dataset with points in a circle"""
    X = []
    y = []
    for i in range(N):
        x = random.uniform(-1.0, 1.0)
        y_val = random.uniform(-1.0, 1.0)
        if x * x + y_val * y_val < 0.5:
            label = 1
        else:
            label = 0
        X.append((x, y_val))
        y.append(label)
    return Graph(N, X, y)

def spiral(N: int = 100) -> Graph:
    """Dataset with points in a spiral pattern"""
    X = []
    y = []
    for i in range(N):
        radius = random.uniform(0, 1)
        angle = random.uniform(0, 4 * math.pi)
        if random.random() > 0.5:
            x = radius * math.cos(angle)
            y_val = radius * math.sin(angle)
            label = 1
        else:
            x = radius * math.cos(angle + math.pi)
            y_val = radius * math.sin(angle + math.pi)
            label = 0
        X.append((x, y_val))
        y.append(label)
    return Graph(N, X, y)

datasets = {'Simple': simple, 'Diag': diag, 'Split': split, 'Xor': xor, 'Circle': circle, 'Spiral': spiral}