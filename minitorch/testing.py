from typing import Callable, Generic, Iterable, Tuple, TypeVar
import minitorch.operators as operators
A = TypeVar('A')

class MathTest(Generic[A]):

    @staticmethod
    def neg(a: A) -> A:
        """Negate the argument"""
        pass

    @staticmethod
    def addConstant(a: A) -> A:
        """Add contant to the argument"""
        pass

    @staticmethod
    def square(a: A) -> A:
        """Manual square"""
        pass

    @staticmethod
    def cube(a: A) -> A:
        """Manual cube"""
        pass

    @staticmethod
    def subConstant(a: A) -> A:
        """Subtract a constant from the argument"""
        pass

    @staticmethod
    def multConstant(a: A) -> A:
        """Multiply a constant to the argument"""
        pass

    @staticmethod
    def div(a: A) -> A:
        """Divide by a constant"""
        pass

    @staticmethod
    def inv(a: A) -> A:
        """Invert after adding"""
        pass

    @staticmethod
    def sig(a: A) -> A:
        """Apply sigmoid"""
        pass

    @staticmethod
    def log(a: A) -> A:
        """Apply log to a large value"""
        pass

    @staticmethod
    def relu(a: A) -> A:
        """Apply relu"""
        pass

    @staticmethod
    def exp(a: A) -> A:
        """Apply exp to a smaller value"""
        pass

    @staticmethod
    def add2(a: A, b: A) -> A:
        """Add two arguments"""
        pass

    @staticmethod
    def mul2(a: A, b: A) -> A:
        """Mul two arguments"""
        pass

    @staticmethod
    def div2(a: A, b: A) -> A:
        """Divide two arguments"""
        pass

    @classmethod
    def _tests(cls) -> Tuple[Tuple[str, Callable[[A], A]], Tuple[str, Callable[[A, A], A]], Tuple[str, Callable[[Iterable[A]], A]]]:
        """
        Returns a list of all the math tests.
        """
        pass

class MathTestVariable(MathTest):
    pass