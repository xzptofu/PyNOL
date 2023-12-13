from typing import Optional, Union

import numpy as np
from pynol.environment.domain import Domain,Ball
from pynol.learner.base import SOGD
from pynol.learner.meta import AdaptMLProd ,AdaNormalHedge
from pynol.learner.models.model import Model
from pynol.learner.schedule.cover import PCGC
from pynol.learner.schedule.schedule import PSchedule
from pynol.learner.schedule.ssp import StepSizeFreeSSP
#-------------------------------------------------
from pynol.learner.specification.surrogate_meta import SurrogateMetaFromBase
from pynol.learner.specification.surrogate_base import Surrogate4RPCBase
from pynol.environment.environment import Environment


class EAPAR(Model):
    """Implementation of Efficient Algorithm for Problem-dependent Adaptive Regret.

        --------------------------
        description to be done
    """

    def __init__(self,
                 domain: Domain,
                 T: int,
                 G: float = None,
                 surrogate: bool = True,
                 loss_threshold: float = 1.,
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        D = 2 * domain.R
        N = int(np.ceil(np.log2(T + 1)))
        ssp = StepSizeFreeSSP(
            SOGD, 
            num_bases=N, 
            domain=Ball(dimension=domain.dimension, radius=domain.R),
            prior=prior, 
            seed=seed
            )
        meta = AdaptMLProd(N=N)
        #meta = AdaNormalHedge(N = N)
        cover = PCGC(N, loss_threshold)
        schedule = PSchedule(ssp, cover)
        scale_factor = 2 * G * D if G is not None else 1
        surrogate_base = Surrogate4RPCBase(scale_factor) if surrogate is True else None
        #surrogate_base = Surrogate4RPCBase() if surrogate is True else None
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
        marker = self.schedule.cover.marker
        self.meta.opt_by_gradient(self.loss_bases, surrogate_loss, np.log(1 + 2 * marker))

        # compute internal optimism of bases
        self.compute_internal_optimism(variables)

        self.t += 1
        return self.x, loss, surrogate_loss