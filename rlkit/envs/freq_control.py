# coding=utf-8
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from gym import spaces
from . import register_env


@register_env('freq_control')
class FreqEnv(object):
    def __init__(self, n_tasks=2, **kwargs):

        # when delta f is stable, terminal is True
        self.terminal = False
        # state
        self.delta_f = 0
        self.roco_f = 0
        self.state_2 = 0
        self.observation_size = 2
        # action
        self.delta_pw = 0
        self.action_size = 1
        self.action_bound = [-1, 1]
        # safe region
        self.delta_f_bound = -0.8
        self.roco_f_lower_bound = -1
        self.roco_f_upper_bound = 1
        # space
        self.action_space = spaces.Box(low=self.action_bound[0], high=self.action_bound[1], shape=(1,))
        self.observation_space = np.array([self.delta_f, self.roco_f])
        self.delta_f_lim = -0.5
        self.t0 = 0
        self.firstIter = 1
        # render
        self.ax = []
        self.ay = []
        # adaptive params
        self.Rg = 0.3
        self.Tg = 6.
        self.Pb = 25.
        self.f0 = 50.
        self.Hc = 4.0 * 1 * self.Pb / self.f0
        self.D = 0.01 * self.Pb
        self.delta_Pl = 1.8

        self.n_task = n_tasks
        self.tasks = self.sample_tasks(self.n_task)

    # action->next state & reward & is_done
    def step(self, action):
        action = np.clip(action, *self.action_bound)[0]

        # # use dynamic
        dt = 0.5
        last_delta_f = self.delta_f

        for _ in range(1):
            self.t0 += dt
            state_1 = self.delta_f
            state_2 = self.state_2
            f = np.array(
                [state_1 - self.D / (2 * self.Hc) * state_1 * dt - state_2 / (2 * self.Hc) * dt - self.delta_Pl / (2 * self.Hc) * dt,
                 state_2 + state_1 / (self.Rg * self.Tg) * dt - state_2 / self.Tg * dt])
            g = np.array([-dt / (2 * self.Hc), 0])

            self.delta_f = f[0] + g[0]*action
            self.state_2 = f[1] + g[1]*action
            s = np.array([self.delta_f, self.state_2], dtype=np.float64)
        self.roco_f = (self.delta_f - last_delta_f) / 0.5

        if self.terminal:
            r = 0
            info = {"done": True}
            self.t0 = 0
        elif self.delta_f < self.delta_f_bound:
            r = -200
            info = {"not_done": True, "stable": False}
        else:
            # Phase II
            if abs(last_delta_f - self.delta_f) <= 0.001 and abs(self.roco_f) <= 0.1:
                if self.delta_f < self.delta_f_lim:
                    r = - 50 * abs(action) - 40
                else:
                    r = - 50 * abs(action) + 10
                info = {"not_done": True, "stable": True}
            # Phase I
            else:
                r = - 50 * abs(action) - self.t0 // 2
                info = {"not_done": True, "stable": False}
        return s, r, self.terminal, info

    def reset(self):
        self.terminal = False
        self.delta_f = 0
        # self.state_2 = 0
        self.roco_f = 1
        self.t0 = 0
        self.ax = []
        self.ay = []


        return np.array([self.delta_f, self.roco_f], dtype=np.float64)

    def seed(self, seed):
        np.random.seed(seed)

    def render(self):
        plt.ion()
        self.ax.append(self.t0)
        self.ay.append(self.delta_f)
        plt.clf()
        plt.xlim(0, 60)
        plt.plot(self.ax, self.ay)
        plt.pause(0.01)
        plt.ioff()

    # for test the env
    def sample_action(self):
        # a = np.random.uniform(self.action_bound[0], self.action_bound[1])
        a = [0]
        return a

    def sample_tasks(self, num_tasks):
        Hc_range_1 = np.random.uniform(3, 4, size=(num_tasks // 3,))
        Hc_range_2 = np.random.uniform(4, 5, size=(num_tasks - num_tasks // 3,))
        Rg_range_1 = np.random.uniform(0.2, 0.3, size=(num_tasks // 3,))
        Rg_range_2 = np.random.uniform(0.2, 0.3, size=(num_tasks - num_tasks // 3,))
        PL_range_1 = np.random.uniform(1.7, 2.2, size=(num_tasks // 3,))
        PL_range_2 = np.random.uniform(2.2, 2.7, size=(num_tasks - num_tasks // 3,))
        tasks = []
        for i in range(num_tasks // 3):
            tasks.append({'Hc': round(Hc_range_1[i], 2), 'Rg': round(Rg_range_1[i], 2), 'PL': round(PL_range_1[i], 2)})
        for i in range(num_tasks - num_tasks // 3):
            tasks.append({'Hc': round(Hc_range_2[i], 2), 'Rg': round(Rg_range_2[i], 2), 'PL': round(PL_range_2[i], 2)})
        return tasks

    def sample_test_tasks(self, num_tasks):
        np.random.seed(1)
        Hc_range = np.random.uniform(3, 5, size=num_tasks)
        Rg_range = np.random.uniform(0.2, 0.3, size=num_tasks)
        PL_range = np.random.uniform(1.7, 2.7, size=num_tasks)
        tasks = []
        for i in range(num_tasks):
            tasks.append({'Hc': round(Hc_range[i], 2), 'Rg': round(Rg_range[i], 2), 'PL': round(PL_range[i], 2)})
        return tasks

    def set_task(self, task):
        self.Hc = task['Hc'] * self.Pb / self.f0
        self.Rg = task['Rg']
        self.delta_Pl = task['PL']


    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()


if __name__ == '__main__':
    # # test
    np.random.seed(1)
    env = FreqEnv()
    state = []
    for ep in range(1):
        s0 = env.reset()
        for t in range(1000):
            a = env.sample_action()
            s, r, done, info = env.step(a)
            state.append(s.tolist())
            if done:
                break
    plt.figure()
    x = np.linspace(0, len(state), len(state))
    plt.plot(x, np.array(state)[:, 0])
    plt.show()


