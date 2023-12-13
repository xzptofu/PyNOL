import os
from pynol.environment.domain import Ball, Simplex, Ellipsoid
from pynol.environment.environment import Environment
from pynol.environment.loss_function import SquareLoss
from pynol.learner.base import OGD
from pynol.learner.models.dynamic.ader import Ader 
from pynol.learner.models.dynamic.ader_1p import Ader_1p
from pynol.learner.models.dynamic.sword import SwordBest
from pynol.online_learning import multiple_online_learning, online_learning
from pynol.utils.data_generator import LinearRegressionGenerator
from pynol.utils.plot import plot
import numpy as np

T, dimension, stage, R, Gamma, scale, seed = 10000, 3, 100, 1, 1, 1 / 2, 0
feature, label = LinearRegressionGenerator().generate_data(
    T, dimension, stage, R, Gamma, seed=seed)
D, r = 2 * R, R
G = scale * D * Gamma**2
C = scale * 1 / 2 * (D * Gamma)**2
L_smooth = Gamma**2

seeds = range(5)
#domain = Ball(dimension=dimension, radius=R)
domain = Ellipsoid(dimension=dimension, E=np.array([1,2,3]), radius=R)
min_step_size, max_step_size = D / (G * T**0.5), D / G
ogd = [
    OGD(domain=domain, step_size=min_step_size, seed=seed) for seed in seeds
]
ader = [
    Ader(
        domain=domain,
        T=T,
        G=G,
        surrogate=False,
        min_step_size=min_step_size,
        max_step_size=max_step_size,
        seed=seed) for seed in seeds
]

aderpp = [
    Ader(
        domain=domain,
        T=T,
        G=G,
        surrogate=True,
        min_step_size=min_step_size,
        max_step_size=max_step_size,
        seed=seed) for seed in seeds
]
ader1p = [
    Ader_1p(
        domain=domain,
        T=T,
        G=G,
        surrogate=True,
        min_step_size=min_step_size,
        max_step_size=max_step_size,
        seed=seed) for seed in seeds
]

sword = [
    SwordBest(
        domain=domain,
        T=T,
        G=G,
        L_smooth=L_smooth,
        min_step_size=min_step_size,
        max_step_size=max_step_size,
        seed=seed) for seed in seeds
]

learners = [sword, ader1p ]
labels = ['Sword', 'Ader_1p']

if __name__ == "__main__":
    loss_func = SquareLoss(feature=feature, label=label, scale=scale)
    env = Environment(func_sequence=loss_func)
    _, loss, _, end_time = multiple_online_learning(T, env, learners)
    if os.path.exists('./results') is False:
        os.makedirs('./results')
    plot(loss, labels, file_path='./results/dynamic_full_info_test.pdf')
    plot(end_time / 1000, labels, file_path='./results/dynamic_full_info_test_time.pdf', y_label='Running time')
