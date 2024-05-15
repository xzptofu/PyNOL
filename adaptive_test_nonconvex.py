import os
import numpy as np
from pynol.environment.domain import Ball, Ellipsoid
from pynol.environment.environment import Environment
from pynol.environment.loss_function import SquareLoss
from pynol.learner.base import SOGD, FTPL
from pynol.learner.models.adaptive.aflh import AFLH
from pynol.learner.models.adaptive.sacs import PSACS, SACS
from pynol.learner.models.adaptive.saol import SAOL
from pynol.learner.models.adaptive.eapar import EAPAR
from pynol.learner.models.adaptive.ftpla import FTPLA
from pynol.online_learning import multiple_online_learning, online_learning
from pynol.utils.data_generator import LinearRegressionGenerator
from pynol.utils.plot import plot

import matplotlib.pyplot as plt

T, dimension, stage, R, Gamma, scale, seed = 500, 1, 5, 1, 1, 1 / 6, 0
feature, label = LinearRegressionGenerator().generate_data(
    T, dimension, stage, R, Gamma, seed=seed)
alive_time_threshold, loss_threshold = 8, 1

seeds = range(5)
domain = Ball(dimension=dimension, radius=R, center=np.array([1.5]))
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

ftpla = [
    FTPLA(domain=domain, T=T, alive_time_threshold=alive_time_threshold, seed=seed)
    for seed in seeds
]

ftpl = [
    FTPL(
        domain=domain,
        step_size = 1 / (dimension *T)**0.5,
        seed=seed) for seed in seeds
]

'''
learners = [sacs, ftpla]
labels = ['SACS', 'FTPLA']
'''
learners = [sogd, ftpl, aflhpp, sacspp, ftpla]
labels = ['SOGD', 'FTPL', 'AFLH++', 'PSACS', 'FTPLA']


from typing import Callable
import autograd.numpy as np
import math
class Poly:
    def __init__(self,
                 para: np.ndarray = None,
                 scale: float = 1.) -> None:
        self.para = para
        self.scale = scale

    def __getitem__(self, t: int) -> Callable[[np.ndarray], float]:
        return lambda x: (x**3 - x**2 - self.para[t]*x + 20) / self.scale

class SinPoly:
    def __init__(self,
                 para: np.ndarray = None,
                 scale: float = 1.) -> None:
        self.para = para
        self.scale = scale

    def __getitem__(self, t: int) -> Callable[[np.ndarray], float]:
        return lambda x: (np.sin((x[0]**2+x[1]**2+x[0]-x[1]+self.para[t]))+1) / self.scale


class Ackley:
    def __init__(self,
                 center: np.ndarray = None,
                 scale: float = 1.) -> None:
        self.center = center
        self.scale = scale

    def __getitem__(self, t: int) -> Callable[[np.ndarray], float]:
        return lambda x: -20.0 * np.exp(-0.2 * np.sqrt(0.5 * ((x[0]-self.center[t,0])**2 + (x[1]-self.center[t,1])**2))) - np.exp(0.5 * (np.cos(2 * np.pi * (x[0]-self.center[t,0])) + np.cos(2 * np.pi * (x[1]-self.center[t,1])))) + 20.0 + np.e


class Rastringin:
    def __init__(self,
                 center: np.ndarray = None,
                 scale: float = 1.) -> None:
        self.center = center
        self.scale = scale

    def __getitem__(self, t: int) -> Callable[[np.ndarray], float]:
        return lambda x: 10 * 2 + (x[0]-self.center[t,0])**2 + (x[1]-self.center[t,1])**2 - 10 * math.cos(2 * np.pi * (x[0]-self.center[t,0])) - 10 * math.cos(2 * np.pi * (x[1]-self.center[t,1]))


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
            return lambda x: ( np.sin(10 * np.pi * x) / (2 * x) + (x - 1)**4 + 1 ) * self.scale
    

if __name__ == "__main__":
    #loss_func = Poly(center_list,5)
    #loss_func = Ackley(center_list,50)
    #loss_func = Ackley(center_list,5)
    #loss_func = SinPoly(np.array([i//10 for i in range(T)]),1)
    #loss_func = SquareLoss(feature=feature, label=label, scale=scale)
    reverse = np.array([0,1,0,1,0])#(np.random.rand(stage,dimension))*2 - 0.5
    reverse_list = np.repeat(reverse,T//stage,axis=0)
    loss_func = GRAMACY_and_LEE(reverse_list, scale)
    env = Environment(func_sequence=loss_func)
    x, loss, _, end_time = multiple_online_learning(T, env, learners)
    if os.path.exists('./results') is False:
        os.makedirs('./results')
    plot(loss, labels, file_path='./results/adaptive_test.pdf')
    #plot(np.squeeze(x), labels,cum=False)
    #plot(end_time / 1000, labels, file_path='./results/adaptive_test_time.pdf', y_label='Running time')