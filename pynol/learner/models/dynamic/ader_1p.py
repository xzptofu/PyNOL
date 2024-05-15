from typing import Optional, Union

import numpy as np
from pynol.environment.domain import Domain, Simplex, Ball
from pynol.learner.base import OGD
from pynol.learner.meta import Hedge
from pynol.learner.models.model import Efficient_Model
from pynol.learner.schedule.schedule import Schedule
from pynol.learner.schedule.ssp import DiscreteSSP
from pynol.learner.specification.surrogate_base import LinearSurrogateBase
from pynol.learner.specification.surrogate_meta import SurrogateMetaFromBase
#-------------------------------------------------
from pynol.environment.environment import Environment
from pynol.learner.specification.surrogate_base import Surrogate4RPCBase
from pynol.environment.domain import Ball


class Ader_1p(Efficient_Model):
    """Implementation of Adaptive Online Learning in Dynamic Environments 
    with One Projection at each round.

    ``Ader_1p`` is an improved version of ``Ader``, who reduces the gradient
    query complexity of each round from :math:`\mathcal{O}(\log T)` to :math:`1`.
    ``Ader_1p`` enjoys a dynamic regret bound of :math:`\mathcal{O}(\sqrt{T(1+P_T)})` 
    and a small-loss dynamic regret bound of :math:`\mathcal{O}(\sqrt{(1 + P_T + F_T)(1 + P_T)})`,
    where :math:`F_T =\sum_{t=1}^T f_t(u_t)` is the cumulative loss of the comparator
    sequence.

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
        https://proceedings.neurips.cc/paper_files/paper/2022/file/4b70484ebef62484e0c8cdd269e482fd-Paper-Conference.pdf
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
        
