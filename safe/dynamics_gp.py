#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import sympy as sp
from cvxopt import matrix
from cvxopt import solvers
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C


class DynamicGP:
    def __init__(self, env):
        self.env = env
        self.GP_model = self.build_GP_model()
        self.GP_model_prev = self.GP_model
        self.firstIter = 1
        self.count = 1

    def build_GP_model(self):
        N = self.env.observation_size
        GP_list = []
        noise = 0.01
        for i in range(N):
            kern = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
            gp = GaussianProcessRegressor(kernel=kern, alpha=noise, n_restarts_optimizer=10)
            GP_list.append(gp)
        self.GP_model = GP_list

    # Get the dynamics of the system from the current time step with the RL action
    def get_dynamics(self, obs, u_rl):
        Rg = self.env.Rg
        Tg = self.env.Tg
        Pb = self.env.Pb
        # f0 = self.env.f0
        Hc = self.env.Hc * 1.05
        D = 0.01 * Pb
        delta_Pl = self.env.delta_Pl * 1.05
        dt = 0.5
        obs = np.squeeze(obs)
        state_1 = obs[0]
        state_2 = obs[1]
        f = np.array(
            [state_1 - D / (2 * Hc) * state_1 * dt - state_2 / (2 * Hc) * dt - (delta_Pl + u_rl) / (2 * Hc) * dt,
             state_2 + state_1 / (Rg * Tg) * dt - state_2 / Tg * dt])
        f_ = np.array([-D / (2 * Hc) * state_1 - state_2 / (2 * Hc) - (delta_Pl + u_rl) / (2 * Hc),
                       state_1 / (Rg * Tg) - state_2 / Tg])
        g = np.array([-dt / (2 * Hc), 0])
        g_ = np.array([-1 / (2 * Hc), 0])
        x = np.array([state_1, state_2])
        return [np.squeeze(f), np.squeeze(g), np.squeeze(x)], [np.squeeze(f_), np.squeeze(g_)]

    # Build barrier function model
    def update_GP_dynamics(self, path):
        N = self.env.observation_size
        X = path['Observation']
        X = X.reshape((len(X), 2))
        U = path['Action']
        L = X.shape[0]
        err = np.zeros((L - 1, N))
        S1 = np.zeros((L - 1, 3))
        S2 = np.zeros((L - 1, 3))
        print("---|update dynamics|---")
        for i in range(L - 1):
            [f, _, _, _], [f_, g_] = self.get_GP_dynamics(X[i, :], U[i])
            next_delta_f = X[i + 1, 0]
            next_state_2 = X[i + 1, 1]
            S1[i, :] = np.array([X[i, 0], f_[0], g_[0]])
            S2[i, :] = np.array([X[i, 1], f_[1], g_[1]])
            err[i, :] = np.array([next_delta_f, next_state_2]) - f
        self.GP_model[0].fit(S1[:], err[:, 0])
        self.GP_model[1].fit(S2[:], err[:, 1])

    def get_GP_dynamics(self, obs, u_rl):
        [f_nom, g, x], [f_, g_] = self.get_dynamics(obs, u_rl)
        f = np.zeros(2)
        [m1, std1] = self.GP_model[0].predict(np.array([x[0], f_[0], g_[0]]).reshape(1, -1), return_std=True)
        [m2, std2] = self.GP_model[1].predict(np.array([x[1], f_[1], g_[1]]).reshape(1, -1), return_std=True)
        f[0] = f_nom[0] + m1
        f[1] = f_nom[1] + m2
        f_[0] += m1
        f_[1] += m2
        return [np.squeeze(f), np.squeeze(g), np.squeeze(x), np.array([np.squeeze(std1), np.squeeze(std2)])], [
            np.squeeze(f_), np.squeeze(g_)]

    def get_GP_dynamics_prev(self, obs, u_rl):
        [f_nom, g, x], [f_, g_] = self.get_dynamics(obs, u_rl)
        f = np.zeros(2)
        [m1, std1] = self.GP_model_prev[0].predict(np.array([x[0], f_[0], g_[0]]).reshape(1, -1), return_std=True)
        [m2, std2] = self.GP_model_prev[1].predict(np.array([x[1], f_[1], g_[1]]).reshape(1, -1), return_std=True)
        f[0] = f_nom[0] + m1
        f[1] = f_nom[1] + m2
        f_[0] += m1
        f_[1] += m2
        return [np.squeeze(f), np.squeeze(g), np.squeeze(x), np.array([np.squeeze(std1), np.squeeze(std2)])], [
            np.squeeze(f_), np.squeeze(g_)]
