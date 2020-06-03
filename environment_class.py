#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    File description: Description of the environment.
"""

# Modified Author: Baekjin Kim (baekjin@umich.edu)
# Copyright (c) 2019, Yoan Russac (yoan.russac@ens.fr)
# License: BSD (3-clause)

# Importation
from arm_class import ArmGaussian
import numpy as np
import pandas as pd


class Environment:
    """
    param:
        - K: Number of arms (int)
        - d: Full dimension of the problem, dimension of the actions (int)
        - theta: d-dimensional vector, key hidden parameter of the problem
    """
    def __init__(self, d, theta, sigma_noise, verbose, param=None, pool_yes=None, pool_no=None):
        self.theta = theta
        self.arms = []
        self.dim = d
        self.verbose = verbose
        self.sigma_noise = sigma_noise
        self.param = param
        self.pool_yes = pool_yes
        self.pool_no = pool_no

    def get_arms(self, k, seed):
        """
        Sample k arms on the sphere
        param:
            - k: number of arms generated
            - unif_min: minimum bound for the uniform distribution
            - unif_max: maximum bound for the uniform distribution

        """
        arm1 = self.pool_yes.sample(n=k, random_state= seed).to_numpy()
        arm2 = self.pool_no.sample(n=k, random_state= seed).to_numpy()
        self.arms = []
        for i in range(k):
            self.arms.append(ArmGaussian(arm1[i]))
            self.arms.append(ArmGaussian(arm2[i]))    
        return self.arms

    def play(self, choice):
        """
        Play arms and return the corresponding rewards
        Choice is an int corresponding to the number of the action
        """
        assert isinstance(choice, np.integer), 'Choice type should be np.int !'
        reward = self.arms[choice].pull(self.theta, self.sigma_noise)
        action_played = self.arms[choice].features
        return reward, action_played

    def get_expected_rewards(self):
        """
        Return the expected payoff of the contextualized arm armIndex
        """
        return [arm.get_expected_reward(self.theta) for arm in self.arms]

    def get_best_arm(self):
        """
        Return the indices of the best arm
        """
        current_rewards = self.get_expected_rewards()  # list of the expected rewards
        assert len(current_rewards) > 0, "Error: No action generated, current_rewards is empty"
        best_arm = np.argmax(current_rewards)  # number of the best arm
        assert isinstance(best_arm, np.integer), "Error: bestArm type should be int"
        best_reward = current_rewards[best_arm]  # current reward for this arm
        return best_arm, best_reward

    def display(self):
        for index, arm in enumerate(self.arms):
            print('===========================')
            print('ARM : %d', index + 1)
            print('arm features: %f', arm.features)
