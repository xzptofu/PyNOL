import os
import numpy as np
from pynol.environment.domain import Ball, Ellipsoid
from pynol.environment.environment import Environment
from pynol.environment.loss_function import SquareLoss
from pynol.learner.base import SOGD
from pynol.learner.models.adaptive.aflh import AFLH
from pynol.learner.models.adaptive.sacs import PSACS, SACS
from pynol.learner.models.adaptive.saol import SAOL
from pynol.learner.models.adaptive.eapar import EAPAR
from pynol.online_learning import multiple_online_learning, online_learning
from pynol.utils.data_generator import LinearRegressionGenerator
from pynol.utils.plot import plot

import matplotlib.pyplot as plt

T, dimension, stage, R, Gamma, scale, seed = 2000, 3, 100, 1, 1, 1 / 2, 0
feature, label = LinearRegressionGenerator().generate_data(
    T, dimension, stage, R, Gamma, seed=seed)
alive_time_threshold, loss_threshold = np.log2(T)**2, 1

seeds = range(5)
#domain = Ball(dimension=dimension, radius=R)
domain = Ellipsoid(dimension=dimension, E=np.array([1,2,3]), radius=R)
sogd = [SOGD(domain=domain, seed=seed) for seed in seeds]
aflh = [AFLH(domain=domain, T=T, surrogate=False, seed=seed) for seed in seeds]
aflhpp = [
    AFLH(domain=domain, T=T, surrogate=True, seed=seed) for seed in seeds
]
saol = [
    SAOL(
        domain=domain,
        T=T,
        alive_time_threshold=alive_time_threshold,
        seed=seed) for seed in seeds
]
sacs = [
    SACS(
        domain=domain,
        T=T,
        alive_time_threshold=alive_time_threshold,
        seed=seed) for seed in seeds
]
sacspp = [
    PSACS(domain=domain, T=T, loss_threshold=loss_threshold, seed=seed)
    for seed in seeds
]

eapar = [
    EAPAR(domain=domain, T=T, loss_threshold=loss_threshold, seed=seed)
    for seed in seeds
]


learners = [eapar, sacspp]
labels = ['EAPAR', 'PSACS']

if __name__ == "__main__":
    loss_func = SquareLoss(feature=feature, label=label, scale=scale)
    env = Environment(func_sequence=loss_func)
    _, loss, _, end_time = multiple_online_learning(T, env, learners)
    if os.path.exists('./results') is False:
        os.makedirs('./results')
    plot(loss, labels, file_path='./results/adaptive_test.pdf')
    plot(end_time / 1000, labels, file_path='./results/adaptive_test_time.pdf', y_label='Running time')