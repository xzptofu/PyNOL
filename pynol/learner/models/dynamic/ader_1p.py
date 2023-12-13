from typing import Optional, Union

import numpy as np
from pynol.environment.domain import Domain, Simplex, Ball
from pynol.learner.base import OGD
from pynol.learner.meta import Hedge
from pynol.learner.models.model import Model
from pynol.learner.schedule.schedule import Schedule
from pynol.learner.schedule.ssp import DiscreteSSP
from pynol.learner.specification.surrogate_base import LinearSurrogateBase
from pynol.learner.specification.surrogate_meta import SurrogateMetaFromBase
#-------------------------------------------------
from pynol.environment.environment import Environment
from pynol.learner.specification.surrogate_base import Surrogate4RPCBase
from pynol.environment.domain import Ball


class Ader_1p(Model):
    """Implementation of Adaptive Online Learning in Dynamic Environments with One Projection at each round.

    Ader is an online algorithm designed for optimizing dynamic regret for
    general convex online functions, which is shown to enjoy
    :math:`\mathcal{O}(\sqrt{T(1+P_T)})` dynamic regret.

    Args:
        domain (Domain): Feasible set for the algorithm.
        T (int): Total number of rounds.
        G (float): Upper bound of gradient.
        surrogate (bool): Whether to use surrogate loss.
        min_step_size (float): Minimal step size for the base-learners. It is
            set as the theory suggests by default.
        max_step_size (float): Maximal step size for the base-learners. It is
            set as the theory suggests by default.
        prior (str, numpy.ndarray, optional): The initial decisions of all
            base-learners are set as `domain(prior=prior, see=seed)` for the
            algorithm.
        seed (int, optional): The initial decisions of all base-learners are set
            as `domain(prior=prior, see=seed)` for the algorithm.

    References:
        
    """

    def __init__(self,
                 domain: Domain,
                 T: int,
                 G: float,
                 surrogate: bool = True,
                 min_step_size: Optional[float] = None,
                 max_step_size: Optional[float] = None,
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        D = 2 * domain.R
        if min_step_size is None:
            min_step_size = D / G * (7 / (2 * T))**0.5
        if max_step_size is None:
            max_step_size = D / G * (7 / (2 * T) + 2)**0.5
        ssp = DiscreteSSP(
            OGD,
            min_step_size,
            max_step_size,
            grid=2,
            domain=Ball(dimension=domain.dimension, radius=domain.R), 
            prior=prior,
            seed=seed)
        schedule = Schedule(ssp)
        # learning rate 
        lr = np.array([1 / (G * D * (t + 1)**0.5) for t in range(T)])
        #N = np.log2(1 + 2 * T / 5)/2 + 1
        #lr = np.array([(np.log(N) / (1 + (G*D)**2 * T))**0.5 for t in range(T)])
        meta = Hedge(prob=Simplex(len(ssp)).init_x(prior='nonuniform'), lr=lr)
        surrogate_base = Surrogate4RPCBase() if surrogate is True else None
        surrogate_meta = SurrogateMetaFromBase() if surrogate is True else None
        self.domain = domain
        super().__init__(
            schedule,
            meta,
            surrogate_base=surrogate_base,
            surrogate_meta=surrogate_meta)
        
    def opt_by_gradient(self, env: Environment):
        """Optimize by the true gradient.

        Args:
            env (Environment): Environment at the current round.

        Returns:
            tuple: tuple contains:
                x (numpy.ndarray): the decision at the current round. \n
                loss (float): the origin loss at the current round. \n
                surrogate_loss (float): the surrogate loss at the current round.
        """
        self.env = env
        variables = vars(self)
        self.x_bases = self.schedule.x_active_bases
        self.y = np.dot(self.meta.prob, self.x_bases)
        # the only projection at each round
        self.x = self.domain.project(self.y)

        if env.full_info:
            loss, surrogate_loss = env.get_loss(self.x)
            self.grad = env.get_grad(self.x)
        else:
            self.perturbation.perturb_x(self.x)
            loss, surrogate_loss = self.perturbation.compute_loss(env)
            self.grad = self.perturbation.construct_grad()

        # construct surrogate loss of environment
        surrogate_func, surrogate_grad = self.surrogate_base.compute_surrogate_base(variables)

        # update bases
        base_env = Environment(func=surrogate_func, grad_func=surrogate_grad)
        if self.surrogate_base is not None:
            base_env.surrogate_func, base_env.surrogate_grad = self.surrogate_base.compute_surrogate_base(
                variables)

        self.loss_bases, self.surrogate_loss_bases = self.schedule.opt_by_gradient(
            base_env)

        # compute surrogate loss of meta #
        if self.surrogate_meta is not None:
            self.loss_bases = self.surrogate_meta.compute_surrogate_meta(
                variables)

        # update meta
        surrogate_loss = surrogate_func(self.y)
        self.meta.opt_by_gradient(self.loss_bases, surrogate_loss)

        # compute internal optimism of bases
        self.compute_internal_optimism(variables)

        self.t += 1
        return self.x, loss, surrogate_loss
