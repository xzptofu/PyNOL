import os
from pynol.environment.domain import Ball, Simplex, Ellipsoid
from pynol.environment.environment import Environment
from pynol.environment.loss_function import SquareLoss
from pynol.learner.base import OGD, FTPL
from pynol.learner.models.dynamic.ader import Ader 
from pynol.learner.models.dynamic.ader_1p import Ader_1p
from pynol.learner.models.dynamic.sword import SwordBest
from pynol.learner.models.dynamic.ftpld import FTPLD 
from pynol.online_learning import multiple_online_learning, online_learning
from pynol.utils.data_generator import LinearRegressionGenerator
from pynol.utils.plot import plot
import numpy as np

T, dimension, stage, R, Gamma, scale, seed = 500, 1, 5, 1, 1, 1 / 6, 0
feature, label = LinearRegressionGenerator().generate_data(
    T, dimension, stage, R, Gamma, seed=seed)
alive_time_threshold = 8 #np.log2(T)**2
D, r = 2 * R, R
G = scale * (10 * np.pi)
C = scale * 1 / 2 * (D * Gamma)**2
L_smooth = Gamma**2

seeds = range(5)
domain = Ball(dimension=dimension, radius=R, center=np.array([1.5]))
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


ftpl = [
    FTPL(
        domain=domain,
        step_size = 1 / (dimension * T)**0.5,
        seed=seed) for seed in seeds
]

ftpld = [
    FTPLD(
        domain=domain,
        T=T,
        G=G,
        alive_time_threshold = alive_time_threshold,
        seed=seed) for seed in seeds
]

learners = [ogd, ftpl, ader, aderpp, sword, ftpld]
labels = ['OGD', 'FTPL', 'Ader', 'Ader++', 'Sword', 'FTPLD']

from typing import Callable
import autograd.numpy as np
class GRAMACY_and_LEE:
    def __init__(self,
                 reserse: np.ndarray = None,
                 scale: float = 1.) -> None:
        #self.bias = bias
        self.scale = scale
        self.reserse = reserse

    def __getitem__(self, t: int) -> Callable[[np.ndarray], float]:
        if self.reserse[t]:
            #return lambda x: ( np.sin(10 * np.pi * (3 - x)) / (2 * (3 - x)) + ((3 - x) - 1 - self.bias[t])**4 + 1 ) * self.scale
            return lambda x: ( np.sin(10 * np.pi * (3 - x)) / (2 * (3 - x)) + ((3 - x) - 1)**4 + 1 ) * self.scale
        else:
            #return lambda x: ( np.sin(10 * np.pi * x) / (2 * x) + (x - 1 - self.bias[t])**4 + 1 ) * self.scale
            return lambda x: ( np.sin(10 * np.pi * x) / (2 * x) + (x - 1)**2 + 1 ) * self.scale
    


if __name__ == "__main__":
    #np.random.seed(2)
    bias = np.array([0,1.5,0,1.5])#(np.random.rand(stage,dimension))*2 - 0.5
    bias_list = np.repeat(bias,T//stage,axis=0)
    reverse = np.array([0,1,0,1,0])#(np.random.rand(stage,dimension))*2 - 0.5
    reverse_list = np.repeat(reverse,T//stage,axis=0)
    #print(bias)
    loss_func = GRAMACY_and_LEE(reverse_list, scale)
    #loss_func = SquareLoss(feature=feature, label=label, scale=scale)
    env = Environment(func_sequence=loss_func)
    x, loss, _, end_time = multiple_online_learning(T, env, learners)

    if os.path.exists('./results') is False:
        os.makedirs('./results')
    plot(loss, labels, file_path='./results/dynamic_test.pdf')
    #plot(np.squeeze(x), labels,cum=False)
    #plot(end_time / 1000, labels, file_path='./results/dynamic_test_time.pdf', y_label='Running time')