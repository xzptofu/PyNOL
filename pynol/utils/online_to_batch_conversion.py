from typing import Union

import numpy as np

from pynol.environment.loss_function import SquareLoss, LogisticLoss
from pynol.environment.environment import Environment

def online_to_batch_conversion(learner, X, y, loss_func = 'SquareLoss', method = None):
    # loss/criterion (default: square loss)
    if loss_func == 'SquareLoss':
        loss_func = SquareLoss(feature = X, label = y, scale = 0.5)
    elif loss_func == 'LogisticLoss':
        loss_func = LogisticLoss(feature = X, label = y)
    else: 
        pass
    env = Environment(func_sequence=loss_func)
    T = X.shape[0]
    if hasattr(learner, 'domain'):
        dimension = learner.domain.dimension
    else:
        dimension = learner.schedule.bases[0].domain.dimension

    if method == None:
        x = np.zeros((T, dimension))
        loss, surrogate_loss = np.zeros(T), np.zeros(T)
        # train for T rounds
        for t in range(T):
            x[t], loss[t], surrogate_loss[t] = learner.opt(env[t])
            # callback
            if t%100 == 0:
                print(f'step: {t} - loss: {loss[t]}')

    elif method == 'anytime':
        x = np.zeros((T, dimension))
        w = np.zeros((T, dimension))
        loss, surrogate_loss = np.zeros(T), np.zeros(T)
        avg_x = np.zeros(dimension)
        # train for T rounds
        for t in range(T):
            # computer running average
            x_submit = avg_x
            x[t], loss[t], surrogate_loss[t] = learner.opt(env[t]) 
            loss[t], _ = env[t].get_loss(x_submit)
            avg_x = np.mean(x[:t+1, :], axis=0)
            w[t] = x_submit
            # callback
            if t%100 == 0:
                print(f'step: {t} - loss: {loss[t]}')

    else:
        raise ValueError("The method is not supportted.")
        
    # return average model (for OCO)
    learner.avg_x = np.mean(x, axis=0)

    # return random model (for more general cases)
    learner.random_x = x[np.random.choice(x.shape[0]), :]
    
    return x, loss, surrogate_loss
