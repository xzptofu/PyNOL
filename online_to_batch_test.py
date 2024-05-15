import os
import numpy as np
from pynol.environment.domain import Ball
from pynol.environment.environment import Environment
from pynol.environment.loss_function import SquareLoss
from pynol.learner.models.dynamic.ader import Ader
from pynol.utils.data_generator import LinearRegressionGenerator
from pynol.utils.plot import plot
from pynol.utils.online_to_batch_conversion import online_to_batch_conversion
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Preprocess dataset 
X, y = fetch_california_housing(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train.reshape(-1,1))
y_test = y_scaler.transform(y_test.reshape(-1,1))

# Prepare model
T, dimension, R, Gamma, scale = X_train.shape[0], X.shape[1], 1, 1, 1 / 2
D, r = 2 * R, R
G = scale * D * Gamma ** 2
C = scale * 1 / 2 * (D * Gamma)**2
L_smooth = Gamma**2
domain = Ball(dimension=dimension, radius=R)
min_step_size, max_step_size = D / (G * T**0.5), D / G 
online_learner = Ader(domain = domain, 
                      T = T, 
                      G = G, 
                      surrogate = False,
                      min_step_size = min_step_size, 
                      max_step_size = max_step_size, 
                      seed = 0)
labels = 'Ader'

if __name__ == "__main__":
    # Train 
    _, train_loss, _ = online_to_batch_conversion(online_learner, X_train, y_train, loss_func='SquareLoss', method='anytime')

    # Test 
    test_loss_func = SquareLoss(feature=X_test, label=y_test, scale=scale)
    test_loss = np.zeros(y_test.shape)
    for idx in range(X_test.shape[0]):
        theta = online_learner.avg_x
        test_loss[idx] = test_loss_func[idx](theta)
    print(f'Tese MSE: {test_loss.mean()}')

    # Plot training curve
    train_loss = np.expand_dims(train_loss, axis = 0)
    labels = [labels]
    plot(train_loss, labels, file_path='./results/online_to_batch.pdf')
