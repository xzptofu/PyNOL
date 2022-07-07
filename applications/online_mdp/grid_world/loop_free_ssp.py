import cvxpy as cp
import numpy as np

from pynol.environment.domain import Domain


class GridWorld(Domain):

    def __init__(self, row, column, actions, prob=1., num_samples=1) -> None:
        self.row = row
        self.column = column
        self.actions = actions
        self.prob = prob
        self.num_states = row * column
        self.index_states = np.arange(self.num_states)
        self.num_actions = len(actions)
        self.index_actions = np.arange(self.num_actions)
        assert self.num_actions > 1 or self.prob == 1.
        self.layer_num = row + column - 1
        self.transition = self.init_transition()
        self.layers = self.state2layer()
        self.l_constraint, self.r_constraint = self.constraint()
        self.dimension = (self.num_states - 1) * self.num_actions
        self.states = np.zeros(num_samples, dtype=int)
        self.end_state = self.num_states - 1
        self.epsilon = 1e-3

    def init_transition(self):
        transition = np.zeros(
            (self.row, self.column, self.num_actions, self.row, self.column))
        for row in range(self.row):
            for column in range(self.column):
                if row == self.row - 1 and column < self.column - 1:
                    transition[row, column, :, row, column + 1] = 1.
                elif row < self.row - 1 and column == self.column - 1:
                    transition[row, column, :, row + 1, column] = 1.
                elif row < self.row - 1 and column < self.column - 1:
                    for i in range(self.num_actions):
                        for action in self.actions:
                            if action == self.actions[i]:
                                transition[row, column, i, row + action[0],
                                           column + action[1]] = self.prob
                            else:
                                transition[row, column, i, row + action[0],
                                           column +
                                           action[1]] = (1 - self.prob) / (
                                               self.num_actions - 1)
                else:
                    transition[row, column, :, :, :] = 0.
        transition = transition.reshape(self.num_states, self.num_actions,
                                        self.num_states)
        return transition

    def state2layer(self):
        layers = ([], [], [])  # layer, row, col
        for row in range(self.row):
            for col in range(self.column):
                layers[0].append(row + col)
                layers[1].append(row)
                layers[2].append(col)
        return layers

    def constraint(self):
        l_state_constraint = np.eye(self.num_states)[:, None, :].repeat(
            self.num_actions, axis=1)
        l_state_constraint = l_state_constraint.reshape(-1, self.num_states).T
        r_state_constraint = self.transition.reshape(-1, self.num_states).T

        l_layer_constraint = np.zeros(
            (self.layer_num, self.row, self.column, self.num_actions))
        l_layer_constraint[self.layers] = 1.
        l_layer_constraint = l_layer_constraint.reshape(self.layer_num, -1)

        l_constraint_1 = (l_state_constraint -
                          r_state_constraint)[1:-1, :-self.num_actions]
        l_constraint_2 = l_layer_constraint[:1, :-self.num_actions]
        l_constraint = np.concatenate((l_constraint_1, l_constraint_2), 0)
        r_constraint_1 = np.zeros(l_constraint_1.shape[0])
        r_constraint_2 = np.ones(l_constraint_2.shape[0])
        r_constraint = np.concatenate((r_constraint_1, r_constraint_2), 0)
        return l_constraint, r_constraint

    def project(self, q):
        q = np.maximum(q, self.epsilon)
        x = cp.Variable(len(q))
        obj = cp.Minimize(cp.sum(cp.kl_div(x, q)))
        constr = [x >= self.epsilon, self.l_constraint @ x == self.r_constraint]
        problem = cp.Problem(obj, constr)
        try:
            problem.solve()
        except Exception:
            problem.solve(solver='SCS', max_iters=200)
        if not problem.status.startswith('optimal'):
            raise RuntimeError('Optimal solution is not found.')
        return np.maximum(x.value, 1e-3)

    def init_x(self, prior=None, seed=None):
        if prior is None:
            np.random.seed(seed)
            decision = np.random.rand(self.dimension)
            return self.project(decision)
        elif prior == 'uniform':
            return self.project(np.ones(self.dimension))
        elif prior == 'minimum':
            x = cp.Variable(self.dimension)
            obj = cp.Minimize(-cp.sum(cp.entr(x)))
            constr = [x >= self.epsilon, self.l_constraint @ x == self.r_constraint]
            problem = cp.Problem(obj, constr)
            problem.solve()
            return x.value
        else:
            raise TypeError(f'{prior} is not defined.')

    def trans(self, state, action):
        prob = self.transition[state, action, :]
        return np.random.choice(self.index_states, p=prob / prob.sum())

    def sample(self, q, loss):
        q = q.reshape(-1, self.num_actions)
        loss = loss.reshape(-1, self.num_actions)
        cum_loss = np.zeros_like(self.states, dtype=float)
        for i in range(len(self.states)):
            while(self.states[i] != self.end_state):
                action = np.random.choice(
                    np.arange(self.num_actions),
                    p=q[self.states[i]] / q[self.states[i]].sum())
                cum_loss[i] += loss[self.states[i], action]
                self.states[i] = self.trans(self.states[i], action)
        self.states = np.zeros_like(self.states, dtype=int)
        return cum_loss
