from typing import Optional, Union

import numpy as np
from pynol.environment.domain import Domain, Simplex
from pynol.learner.base import FTPL
from pynol.learner.meta import Hedge
from pynol.learner.models.model import Randomized_Model
from pynol.learner.schedule.schedule import Schedule
from pynol.learner.schedule.ssp import DiscreteSSP
from pynol.learner.schedule.cover import DGC, FullCover

class FTPLD(Randomized_Model):
    """Implementation of Follow the Perturbed Leader with Dynamic Regret.

    ``FTPLD`` is an online algorithm designed for optimizing dynamic regret for
    general non-convex online functions, which is shown to enjoy
    :math:`\mathcal{O}(T^\\frac{2}{3}(1+V_T)^\\frac{1}{3})` dynamic regret.

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

    """

    def __init__(self,
                 domain: Domain,
                 T: int,
                 G: float,
                 alive_time_threshold: Optional[int] = None,
                 prior: Optional[Union[list, np.ndarray]] = None,
                 seed: Optional[int] = None):
        D = 2 * domain.R
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
        cover = DGC(N, alive_time_threshold)
        lr = np.array([(8* 5 * np.log(N))**0.5 / (domain.dimension * G * D * (T + 1)**0.5) for t in range(T)])
        meta = Hedge(prob=Simplex(len(ssp)).init_x(prior='uniform'), lr=lr)
        schedule = Schedule(ssp, cover)
        super().__init__(schedule, meta)
