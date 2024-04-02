from typing import Optional, Union

import numpy as np
from pynol.environment.domain import Domain,Ball
from pynol.learner.base import SOGD
from pynol.learner.meta import AdaptMLProd ,AdaNormalHedge
from pynol.learner.models.model import Efficient_Model
from pynol.learner.schedule.cover import PCGC
from pynol.learner.schedule.schedule import PSchedule, Efficient_PSchedule
from pynol.learner.schedule.ssp import StepSizeFreeSSP
#-------------------------------------------------
from pynol.learner.specification.surrogate_meta import SurrogateMetaFromBase
from pynol.learner.specification.surrogate_base import Surrogate4RPCBase
from pynol.environment.environment import Environment


class EAPAR(Efficient_Model):
    """Implementation of Efficient Algorithm for Problem-dependent Adaptive Regret.

    ``EAPAR`` is an efficient model that requires :math:`1` gradient query complexity 
    of each round and enjoys a small-loss adaptive regret bound of :math:'\mathcal{O}
    (\min\{\sqrt{F_I\log F_I\log F_T},\sqrt{|I|\log T}\})', where :math:`F_T =\sum_
    {t=1}^T f_t(u_t)` is the cumulative loss of the comparator sequence.
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
        schedule = Efficient_PSchedule(ssp, cover)
        # rescale the loss function 
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

    def meta_opt_by_gradient(self, surrogate_loss):
        marker = self.schedule.cover.marker
        self.meta.opt_by_gradient(self.loss_bases, surrogate_loss, np.log(1 + 2 * marker))

    def schedule_opt_by_gradient(self, base_env, loss):
        return self.schedule.opt_by_gradient(base_env, loss)
