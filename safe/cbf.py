#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import math
import numpy as np
from cvxopt import matrix
from cvxopt import solvers


class CBFController():
    def __init__(self, env):
        self.env = env

    # Build barrier function model
    def build_barrier(self):
        N = self.env.action_size
        self.P = matrix(np.diag([1., 1e24]), tc='d')
        self.q = matrix(np.zeros(N + 1))

        self.H1 = np.array([1, 0])
        self.H2 = np.array([self.env.D, 1])
        self.H3 = np.array([1, 0])
        self.F1 = 0.8
        self.F2 = self.env.delta_Pl - 8 * self.env.Hc
        self.F3 = 0.5

    # Get compensatory action based on satisfaction of barrier function
    def control_barrier(self, barrier_form, u_rl, f, g, x, std):

        # Set up Quadratic Program to satisfy the Control Barrier Function
        kd = 0.1
        # ZCBF
        if barrier_form == "ZCBF":
            nominal_delta_f = np.array(f[0])
            if abs(self.env.roco_f) < 0.1:
                try:
                    nominal_error = np.clip(nominal_delta_f + 0.5, -np.pi / 2, np.pi / 2)[0]
                except:
                    nominal_error = np.clip(nominal_delta_f + 0.5, -np.pi / 2, np.pi / 2)
                gamma = 1 * np.exp(2 * np.tan(nominal_error))
                G = np.array([[-np.dot(self.H1, g), -np.dot(self.H3, g), 1., -1., g[0], -g[0]], [-1., -1., 0, 0, 0, 0]])
                G = np.transpose(G)
                h = np.array([gamma * self.F1 + np.dot(self.H1, f) + np.dot(self.H1, g) * u_rl + gamma * np.dot(self.H1, x) - kd * np.dot( np.abs(self.H1), std),
                              gamma * self.F3 + np.dot(self.H3, f) + np.dot(self.H3, g) * u_rl + gamma * np.dot(self.H3, x) - kd * np.dot( np.abs(self.H3), std),
                              -u_rl + self.env.action_bound[1],
                              u_rl + self.env.action_bound[1],
                              -f[0] - g[0] * u_rl + self.env.roco_f_upper_bound,
                              f[0] + g[0] * u_rl - self.env.roco_f_lower_bound], dtype=object)

            else:
                try:
                    nominal_error = np.clip(nominal_delta_f + 0.8, -np.pi / 2, np.pi / 2)[0]
                except:
                    nominal_error = np.clip(nominal_delta_f + 0.8, -np.pi / 2, np.pi / 2)
                gamma = 1 * np.exp(2 * np.tan(nominal_error))
                G = np.array([[-np.dot(self.H1, g), -np.dot(self.H3, g), 1., -1.], [-1., -1., 0, 0]])
                G = np.transpose(G)
                h = np.array([gamma * self.F1 + np.dot(self.H1, f) + np.dot(self.H1, g) * u_rl + gamma * np.dot(self.H1, x) - kd * np.dot( np.abs(self.H1), std),
                              gamma * self.F1 + np.dot(self.H1, f) + np.dot(self.H1, g) * u_rl + gamma * np.dot(self.H1, x) - kd * np.dot( np.abs(self.H1), std),
                              -u_rl + self.env.action_bound[1],
                              u_rl + self.env.action_bound[1]], dtype=object)

        # RCBF
        elif barrier_form == "RCBF":
            nominal_delta_f = np.array(f[0])
            if abs(self.env.roco_f) < 0.1:
                try:
                    nominal_error = np.clip(nominal_delta_f + 0.5, -1, 1)[0]
                except:
                    nominal_error = np.clip(nominal_delta_f + 0.5, -1, 1)
                gamma = 2 * np.exp(2 * np.tan(nominal_error))
                G = np.array([[-np.dot(self.H1, g), -np.dot(self.H2, g), 1., -1., g[0], -g[0]], [-1., -1., 0, 0, 0, 0]])
                G = np.transpose(G)
                h = np.array([gamma * (np.dot(self.H1, x) + self.F1 + 1) * (np.dot(self.H1, x) + self.F1) / math.log((1 + 1 / max(1e-5, (np.dot(self.H1, x) + self.F1))), math.e) + np.dot(self.H1, f) + np.dot(self.H1, g) * u_rl - kd * np.dot( np.abs(self.H1), std),
                              gamma * (np.dot(self.H3, x) + self.F3 + 1) * (np.dot(self.H3, x) + self.F3) / math.log( (1 + 1 / max(1e-5, (np.dot(self.H3, x) + self.F3))), math.e) + np.dot(self.H3, f) + np.dot( self.H3, g) * u_rl - kd * np.dot(np.abs(self.H3), std),
                              -u_rl + self.env.action_bound[1],
                              u_rl + self.env.action_bound[1],
                              -f[0] - g[0] * u_rl + self.env.roco_f_upper_bound,
                              f[0] + g[0] * u_rl - self.env.roco_f_lower_bound], dtype=object)
            else:
                try:
                    nominal_error = np.clip(nominal_delta_f + 0.8, -1, 1)[0]
                except:
                    nominal_error = np.clip(nominal_delta_f + 0.8, -1, 1)
                gamma = 2 * np.exp(2 * np.tan(nominal_error))
                G = np.array([[-np.dot(self.H1, g), -np.dot(self.H2, g), 1., -1.], [-1., -1., 0, 0]])
                G = np.transpose(G)
                h = np.array([gamma * (np.dot(self.H1, x) + self.F1 + 1) * (np.dot(self.H1, x) + self.F1) / math.log((1 + 1 / max(1e-5, (np.dot(self.H1, x) + self.F1))), math.e) + np.dot(self.H1, f) + np.dot(self.H1,g) * u_rl - kd * np.dot(np.abs(self.H1), std),
                              gamma * (np.dot(self.H3, x) + self.F1 + 1) * (np.dot(self.H3, x) + self.F1) / math.log((1 + 1 / max(1e-5, (np.dot(self.H3, x) + self.F1))), math.e) + np.dot(self.H3,f) + np.dot(self.H3, g) * u_rl - kd * np.dot(np.abs(self.H3), std),
                              -u_rl + self.env.action_bound[1],
                              u_rl + self.env.action_bound[1]], dtype=object)
        h = np.squeeze(h).astype(np.double)
        # Convert numpy arrays to cvx matrices to set up QP
        G = matrix(G, tc='d')
        h = matrix(h, tc='d')

        solvers.options['show_progress'] = False
        sol = solvers.qp(self.P, self.q, G, h)
        u_bar = sol['x']
        # print(sol['s'])
        if abs(u_bar[0]) < 1e-5:
            u_bar[0] = 0
        if (np.add(np.squeeze(u_rl), np.squeeze(u_bar[0])) - 0.001 >= self.env.action_bound[1]):
            u_bar[0] = self.env.action_bound[1] - u_rl
            print("Error in QP")
        elif (np.add(np.squeeze(u_rl), np.squeeze(u_bar[0])) + 0.001 <= self.env.action_bound[0]):
            u_bar[0] = self.env.action_bound[0] - u_rl
            print("Error in QP")
        else:
            u_bar[0] = round(u_bar[0], 3)

        return np.expand_dims(np.array(u_bar[0]), 0), gamma

    def control_barrier_test(self, barrier_form, alpha, u_rl, f, g, x, std):
        u_rl = u_rl[0]
        # Set up Quadratic Program to satisfy the Control Barrier Function
        kd = 0.5
        # ZCBF
        if barrier_form == "ZCBF":
            nominal_delta_f = np.array(f[0])
            if abs(self.env.roco_f) < 0.01:
                try:
                    nominal_error = np.clip(nominal_delta_f + 0.5, -np.pi / 2, np.pi / 2)[0]
                except:
                    nominal_error = np.clip(nominal_delta_f + 0.5, -np.pi / 2, np.pi / 2)
                gamma = alpha[0] * np.exp(alpha[1] * np.tan(nominal_error))
                G = np.array([[-np.dot(self.H1, g), -np.dot(self.H3, g), 1., -1., g[0], -g[0]], [-1., -1., 0, 0, 0, 0]])
                G = np.transpose(G)
                h = np.array([gamma * self.F1 + np.dot(self.H1, f) + np.dot(self.H1, g) * u_rl + gamma * np.dot(self.H1, x) - kd * np.dot(np.abs(self.H1), std),
                              gamma * self.F3 + np.dot(self.H3, f) + np.dot(self.H3, g) * u_rl + gamma * np.dot(self.H3,x) - kd * np.dot( np.abs(self.H3), std),
                              -u_rl + self.env.action_bound[1],
                              u_rl + self.env.action_bound[1],
                              -f[0] - g[0] * u_rl + self.env.roco_f_upper_bound,
                              f[0] + g[0] * u_rl - self.env.roco_f_lower_bound], dtype=object)
            else:
                try:
                    nominal_error = np.clip(nominal_delta_f + 0.8, -np.pi / 2, np.pi / 2)[0]
                except:
                    nominal_error = np.clip(nominal_delta_f + 0.8, -np.pi / 2, np.pi / 2)
                gamma = alpha[2] * np.exp(alpha[3] * np.tan(nominal_error))
                G = np.array([[-np.dot(self.H1, g), -np.dot(self.H3, g), 1., -1.], [-1., -1., 0, 0]])
                G = np.transpose(G)
                h = np.array([gamma * self.F1 + np.dot(self.H1, f) + np.dot(self.H1, g) * u_rl + gamma * np.dot(self.H1,x) - kd * np.dot(np.abs(self.H1), std),
                              gamma * self.F1 + np.dot(self.H1, f) + np.dot(self.H1, g) * u_rl + gamma * np.dot(self.H1,x) - kd * np.dot(np.abs(self.H1), std),
                              -u_rl + self.env.action_bound[1],
                              u_rl + self.env.action_bound[1]], dtype=object)

        # RCBF
        elif barrier_form == "RCBF":
            nominal_delta_f = np.array(f[0] + g[0] * u_rl)
            if abs(self.env.roco_f) < 0.1:
                try:
                    nominal_error = np.clip(nominal_delta_f + 0.5, -np.pi / 2, np.pi / 2)[0]
                except:
                    nominal_error = np.clip(nominal_delta_f + 0.5, -np.pi / 2, np.pi / 2)
                gamma = alpha[0] * np.exp(alpha[1] * np.tan(nominal_error))
                G = np.array([[-np.dot(self.H1, g), -np.dot(self.H2, g), 1., -1., g[0], -g[0]], [-1., -1., 0, 0, 0, 0]])
                G = np.transpose(G)
                h = np.array([gamma * (np.dot(self.H1, x) + self.F1 + 1) * (np.dot(self.H1, x) + self.F1) / math.log((1 + 1 / max(1e-5, (np.dot(self.H1, x) + self.F1))), math.e) + np.dot(self.H1, f) + np.dot(self.H1,g) * u_rl - kd * np.dot(np.abs(self.H1), std),
                              gamma * (np.dot(self.H3, x) + self.F3 + 1) * (np.dot(self.H3, x) + self.F3) / math.log((1 + 1 / max(1e-5, (np.dot(self.H3, x) + self.F3))), math.e) + np.dot(self.H3,f) + np.dot(self.H3, g) * u_rl - kd * np.dot(np.abs(self.H3), std),
                              -u_rl + self.env.action_bound[1],
                              u_rl + self.env.action_bound[1],
                              -f[0] - g[0] * u_rl + self.env.roco_f_upper_bound,
                              f[0] + g[0] * u_rl - self.env.roco_f_lower_bound], dtype=object)
            else:
                try:
                    nominal_error = np.clip(nominal_delta_f + 0.8, -np.pi / 2, np.pi / 2)[0]
                except:
                    nominal_error = np.clip(nominal_delta_f + 0.8, -np.pi / 2, np.pi / 2)
                gamma = alpha[2] * np.exp(alpha[3] * np.tan(nominal_error))
                G = np.array([[-np.dot(self.H1, g), -np.dot(self.H2, g), 1., -1.], [-1., -1., 0, 0]])
                G = np.transpose(G)
                h = np.array([gamma * (np.dot(self.H1, x) + self.F1 + 1) * (np.dot(self.H1, x) + self.F1) / math.log((1 + 1 / max(1e-5, (np.dot(self.H1, x) + self.F1))), math.e) + np.dot(self.H1, f) + np.dot(self.H1,g) * u_rl - kd * np.dot(np.abs(self.H1), std),
                              gamma * (np.dot(self.H3, x) + self.F1 + 1) * (np.dot(self.H3, x) + self.F1) / math.log((1 + 1 / max(1e-5, (np.dot(self.H3, x) + self.F1))), math.e) + np.dot(self.H3,f) + np.dot(self.H3, g) * u_rl - kd * np.dot(np.abs(self.H3), std),
                              -u_rl + self.env.action_bound[1],
                              u_rl + self.env.action_bound[1]], dtype=object)

        h = np.squeeze(h).astype(np.double)

        G = matrix(G, tc='d')
        h = matrix(h, tc='d')

        solvers.options['show_progress'] = False
        sol = solvers.qp(self.P, self.q, G, h)

        u_bar = sol['x']

        if abs(u_bar[0]) < 1e-5:
            u_bar[0] = 0
        if (np.add(np.squeeze(u_rl), np.squeeze(u_bar[0])) - 0.001 >= self.env.action_bound[1]):
            u_bar[0] = self.env.action_bound[1] - u_rl
            print("Error in QP")
        elif (np.add(np.squeeze(u_rl), np.squeeze(u_bar[0])) + 0.001 <= self.env.action_bound[0]):
            u_bar[0] = self.env.action_bound[0] - u_rl
            print("Error in QP")
        else:
            pass
        return np.expand_dims(np.array(u_bar[0]), 0), gamma
