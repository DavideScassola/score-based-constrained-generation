from abc import ABC, abstractclassmethod

import torch

LOG_EPS = 1e-6
STABLE_PRODUCT_EPS = 1e-2
log_logistic = torch.nn.LogSigmoid()


class RealLogic(ABC):
    @staticmethod
    def greater(
        x1: torch.Tensor, x2: torch.Tensor | float, grad: float = 1.0
    ) -> torch.Tensor:
        return torch.where(x1 > x2, 0.0, x1 - x2) * grad

    @staticmethod
    def smaller(
        x1: torch.Tensor, x2: torch.Tensor | float, grad: float = 1.0
    ) -> torch.Tensor:
        return torch.where(x1 < x2, 0.0, x2 - x1) * grad

    @staticmethod
    def equal(
        x1: torch.Tensor, x2: torch.Tensor | float, k: float = 1.0
    ) -> torch.Tensor:
        return -k * torch.square(x1 - x2)

    @classmethod
    def in_(
        cls,
        x: torch.Tensor,
        inf: torch.Tensor | float,
        sup: torch.Tensor | float,
        grad: float = 1.0,
    ) -> torch.Tensor:
        return cls.and_(cls.greater(x, inf, grad), cls.smaller(x, sup, grad))  # type: ignore

    @abstractclassmethod
    def and_(cls, *x: torch.Tensor) -> torch.Tensor:  # type: ignore
        pass

    @abstractclassmethod
    def or_(cls, *x: torch.Tensor) -> torch.Tensor:  # type: ignore
        pass

    @abstractclassmethod
    def implies(cls, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return cls.or_(cls.not_(x1), x2)

    @classmethod
    def not_(cls, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        pass


class Product(RealLogic):
    """
    Inspired from Badreddine, et al. (2020). Logic Tensor Networks, https://arxiv.org/abs/2012.13635

    For the conjunction and negation
    """

    DEFAULT_GRAD = 50.0

    @staticmethod
    def greater(
        x1: torch.Tensor, x2: torch.Tensor | float, grad: float = DEFAULT_GRAD
    ) -> torch.Tensor:
        return log_logistic(grad * (x1 - x2))  # type: ignore

    @staticmethod
    def smaller(
        x1: torch.Tensor, x2: torch.Tensor | float, grad: float = DEFAULT_GRAD
    ) -> torch.Tensor:
        return log_logistic(grad * (x2 - x1))  # type: ignore

    @classmethod
    def and_(cls, *x: torch.Tensor) -> torch.Tensor:
        """
        If multiple tensors are provided, the element wise 'and' is returned
        If a single tensor is given, then the logical 'and' reduces all dimensions except the first
        """
        if len(x) == 1:
            return x[0].flatten(1).sum(1)
        return torch.stack(x).sum(0)

    @classmethod
    def or_(cls, *x: torch.Tensor, eps=LOG_EPS) -> torch.Tensor:
        if len(x) == 1:
            return x[0]
        if len(x) == 2:
            return torch.log(
                torch.exp(x[0]) + torch.exp(x[1]) - torch.exp(x[0] + x[1]) + eps
            )
        return cls.or_(cls.or_(x[0], x[1]), *x[2:])

    @classmethod
    def not_(cls, x: torch.Tensor) -> torch.Tensor:
        """
        This doesn't work numerically, very hard problem to solve
        """
        return torch.log(1 - torch.exp(x) + LOG_EPS)

    @classmethod
    def in_(
        cls,
        x: torch.Tensor,
        inf: torch.Tensor | float,
        sup: torch.Tensor | float,
        grad: float = DEFAULT_GRAD,
    ) -> torch.Tensor:
        return cls.and_(cls.greater(x, inf, grad), cls.smaller(x, sup, grad))  # type: ignore

    @classmethod
    def equal(
        cls, x1: torch.Tensor, x2: torch.Tensor | float, grad: float = DEFAULT_GRAD
    ) -> torch.Tensor:
        return cls.in_(x1, inf=x2, sup=x2, grad=grad)


class Minimum(RealLogic):
    """
    Inspired from Badreddine, et al. (2020). Logic Tensor Networks, https://arxiv.org/abs/2012.13635
    """

    @classmethod
    def and_(cls, *x: torch.Tensor) -> torch.Tensor:
        return torch.stack(x).amin(0)

    @classmethod
    def or_(cls, *x: torch.Tensor) -> torch.Tensor:
        return torch.stack(x).amax(0)


class Classic(RealLogic):
    """
    Safe logic that I used at the beginning (for aggreating the logical 'and' I used the sum instead, in order to avoid 'single-passing')
    """

    @classmethod
    def and_(cls, *x: torch.Tensor) -> torch.Tensor:
        """
        If multiple tensors are provided, the element wise 'and' is returned
        If a single tensor is given, then the logical 'and' reduces all dimensions except the first
        """
        if len(x) == 1:
            return x[0].flatten(1).sum(1)
        return torch.stack(x).sum(0)

    @classmethod
    def or_(cls, *x: torch.Tensor) -> torch.Tensor:
        return torch.stack(x).amax(0)

    @classmethod
    def implies(cls, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.where(x1 >= 0.0, x2, 0.0)

    @classmethod
    def not_(cls, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Not implemented, use negation only on atoms")


class StableProduct(Product):
    """
    Inspired from Badreddine, et al. (2020). Logic Tensor Networks, https://arxiv.org/abs/2012.13635

    Product logic with numerical corrections in order to avoid gradient issues, but in this case this could be unnecessary since we consider gradient of the log.
    """

    @staticmethod
    def pi0(x: torch.Tensor) -> torch.Tensor:
        """This transformation avoids a value of 0"""
        return torch.log((1 - STABLE_PRODUCT_EPS) * torch.exp(x) + STABLE_PRODUCT_EPS)

    @staticmethod
    def pi1(x: torch.Tensor) -> torch.Tensor:
        """This transformation avoids a value of 1"""
        return torch.log((1 - STABLE_PRODUCT_EPS) * torch.exp(x))

    @classmethod
    def and_(cls, *x: torch.Tensor) -> torch.Tensor:
        return super().and_(*(StableProduct.pi0(xi) for xi in x))

    @classmethod
    def or_(cls, *x: torch.Tensor) -> torch.Tensor:
        return super().or_(*(StableProduct.pi1(xi) for xi in x))

    @classmethod
    def implies(cls, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return cls.implies(StableProduct.pi0(x1), StableProduct.pi1(x2))
