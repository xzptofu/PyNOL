from abc import ABC, abstractmethod

import numpy as np


class SurrogateBase(ABC):
    """The abstract class defines the surrogate loss functions and surrogate
    gradient (if possible) for base-learners."""

    def __init__(self):
        pass

    @abstractmethod
    def compute_surrogate_base(self, variables):
        """Compute the surrogate loss functions and surrogate
        gradient (if possible) for base-learners."""
        raise NotImplementedError()


class LinearSurrogateBase(SurrogateBase):
    """The class defines the linear surrogate loss function for base-learners."""

    def __init__(self):
        pass

    def compute_surrogate_base(self, variables):
        """Compute the surrogate loss functions and surrogate
        gradient (if possible) for base-learners.

        Replace original convex function :math:`f_t(x)` with

        .. math::

            f'_t(x)=\langle \\nabla f_t(x_t),x - x_t \\rangle,

        for all base-learners, where :math:`x_t` is the submitted decision at
        round :math:`t`. Since the gradient of any decision for the linear
        function is :math:`\\nabla f_t(x_t)`, this method will return it also to
        reduce the gradient query complexity for base-learners.

        Args:
            variables (dict): intermediate variables of the learning process at
                current round.

        Returns:
            tuple: tuple contains:
                surrogate_func (Callable): Surrogate function for base-learners. \n
                surrogate_grad (numpy.ndarray): Surrogate gradient for base-learners.
        """
        return lambda x: np.dot(x - variables['x'], variables['grad']), variables['grad']


class InnerSurrogateBase(SurrogateBase):
    """The class defines the inner surrogate loss function for base-learners."""

    def __init__(self):
        pass

    def compute_surrogate_base(self, variables):
        """Compute the surrogate loss functions and surrogate
        gradient (if possible) for base-learners.

        Replace original convex function :math:`f_t(x)` with

        .. math::

            f'_t(x)=\langle \\nabla f_t(x_t), x \\rangle,

        for all base-learners, where :math:`x_t` is the submitted decision at
        round :math:`t`. Since the gradient of any decision for the inner
        function is :math:`\\nabla f_t(x_t)`, this method will return it also to
        reduce the gradient query complexity for base-learners.

        Args:
            variables (dict): intermediate variables of the learning process at
                current round.
        Returns:
            tuple: tuple contains:
                surrogate_func (Callable): Surrogate function for base-learners. \n
                surrogate_grad (numpy.ndarray): Surrogate gradient for base-learners.
        """
        return lambda x: np.dot(x, variables['grad']), variables['grad']


class Surrogate4RPCBase(SurrogateBase):
    """The class defines the surrogate loss function for reducing the projection complexity."""

    def __init__(self, scale_factor : float = 1):
        self.scale_factor = scale_factor

    def compute_surrogate_base(self, variables):
        """Compute the surrogate loss functions and surrogate
        gradient (if possible) for base-learners.

        Replace original convex function :math:`f_t(x)` with

        .. math::

            f'_t(y)= \\langle f_t(x_t),y \\rangle 
                - \\mathbb{1}_{\\{\langle f_t(x_t), v_t \\rangle<0\\} } \\cdot \\langle f_t(x_t), v_t \\rangle \\cdot S_{\mathcal{X}}(y),

        Args:
            variables (dict): intermediate variables of the learning process at
                current round.

        Returns:
            tuple: tuple contains:
                surrogate_func (Callable): Surrogate function for base-learners. \n
                surrogate_grad (numpy.ndarray): Surrogate gradient for base-learners.
        """
        eps = 1e-8
        if (abs(variables['y'] - variables['x']) > eps).any():
            v_t = (variables['y'] - variables['x'])/np.linalg.norm((variables['y'] - variables['x']))
        else:
            v_t = np.zeros(variables['domain'].dimension)
        if np.dot(variables['grad'], v_t) >= 0 :
            grad = variables['grad']
        else:
            grad = variables['grad'] - np.dot(variables['grad'], v_t) * v_t
        return lambda y: np.dot(grad, y) / self.scale_factor  , grad / self.scale_factor