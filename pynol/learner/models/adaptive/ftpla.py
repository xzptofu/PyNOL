from typing import Optional, Union

import numpy as np
from pynol.environment.domain import Domain
from pynol.learner.base import FTPL
from pynol.learner.meta import AdaNormalHedge
from pynol.learner.models.model import Model
from pynol.learner.schedule.cover import GC
from pynol.learner.schedule.schedule import Schedule
from pynol.learner.schedule.ssp import DiscreteSSP


class FTPLA(Model):
    """Implementation of Follow the Perturbed Leader with Adaptive Regret.

    ``FTPLA`` is an online algorithm designed for optimizing dynamic regret for
    general non-convex online functions, which attains
    :math:`\mathcal{O}(\sqrt{(s-r) \log T})` strongly adaptive regret.

    Args:
        domain (Domain): Feasible set for the algorithm.
        T (int): Total number of rounds.
        alive_time_threshold (int, optional): Minimal alive time for base-learners.
            All base-learners whose alive time are less than ``alive_time_threshold``
            will not be activated.
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
                 alive_time_threshold: Optional[int] = None,
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        N = int(np.ceil(np.log2(T + 1)))
        min_lmbd = 1 / (domain.dimension * T)**0.5
        max_lmbd = 1 / domain.dimension**0.5
        ssp = DiscreteSSP(FTPL, 
            min_step_size=min_lmbd,
            max_step_size=max_lmbd,
            grid=2**0.5,
            domain=domain, 
            prior=prior, 
            seed=seed)
        list.reverse(ssp.bases) # Reverse the bases list such that the 'lmbd' is in descending order.
        cover = GC(N, alive_time_threshold)
        meta = AdaNormalHedge(N=N)
        schedule = Schedule(ssp, cover)
        super().__init__(schedule, meta)