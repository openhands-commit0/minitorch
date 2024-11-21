from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple
from typing_extensions import Protocol

def central_difference(f: Any, *vals: Any, arg: int=0, epsilon: float=1e-06) -> Any:
    """
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \\ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \\ldots, x_{n-1})$
    """
    vals_plus = list(vals)
    vals_minus = list(vals)
    vals_plus[arg] = vals[arg] + epsilon
    vals_minus[arg] = vals[arg] - epsilon
    return (f(*vals_plus) - f(*vals_minus)) / (2.0 * epsilon)
variable_count = 1

class Variable(Protocol):
    """A variable in a computation graph."""

    def accumulate_derivative(self, x: Any) -> None:
        """Add `x` to the derivative accumulated on this variable."""
        pass

    def is_constant(self) -> bool:
        """Is this a constant variable?"""
        pass

    def is_leaf(self) -> bool:
        """Is this a leaf variable?"""
        pass

    def parents(self) -> Iterable["Variable"]:
        """Get the parents of this variable."""
        pass

def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = set()
    order = []

    def visit(var: Variable) -> None:
        if var in visited or var.is_constant():
            return
        visited.add(var)
        for parent in var.parents():
            visit(parent)
        order.insert(0, var)

    visit(variable)
    return order

def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    queue = [(variable, deriv)]
    derivatives = {}

    while queue:
        var, d = queue.pop(0)
        if var.is_leaf():
            var.accumulate_derivative(d)
        else:
            for parent in var.parents():
                if parent not in derivatives:
                    derivatives[parent] = 0.0
                derivatives[parent] += d
                queue.append((parent, derivatives[parent]))

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """
    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if not self.no_grad:
            self.saved_values = values