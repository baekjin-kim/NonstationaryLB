# -*- coding: utf-8 -*-

"""
    Simulator class
"""

# Modified Author: Baekjin Kim (baekjin@umich.edu)
# Copyright (c) 2019, Yoan Russac (yoan.russac@ens.fr)
# License: BSD (3-clause)

# Importation
import numpy as np
import multiprocessing
from tqdm import tqdm
import time

class Simulator(object):
    """
    Simulator of stochastic games.
    param:
        - MAB: list, List of arms.
        - policies: list, List of policies to test.
        - K: int, Number of items (arms) in the pool.
        - d: int, Dimension of the problem
        - T: Number of steps in each round
    """
    def __init__(self, mab, theta, policies, k, d, steps, bp_dico, verbose):
        """"
        global init function, initializing all the internal parameters that are fixed
        """
        self.policies = policies
        self.mab = mab
        self.theta = theta
        self.steps = steps
        self.d = d
        self.k = k
        self.verbose = verbose
        self.bp_dico = bp_dico
        if self.verbose:
            print("real theta is ", self.mab.theta)

    def run(self, steps, n_mc, q, n_scatter, t_saved=None):
        """
        Runs an experiment with steps points and n_mc Monte Carlo repetition
        param:
            - steps: Number of steps for an experiment (int)
            - n_mc: Total number of Monte Carlo experiment (int)
            - q: Quantile (int). ex: q=5%
            - n_scatter: Frequency of the plot of the estimate for the scatter plot (only in 2D problems)
            - t_saved: Trajectory of points saved to store fewer than steps points on a trajectory.
                        (numpy array ndim = 1)
        """
        if t_saved is None:
            t_saved = [i for i in range(steps)]
        cum_regret = dict()
        n_sub = np.size(t_saved)  # Number of points saved for each trajectory
        avg_regret = dict()
        q_regret = dict()
        up_q_regret = dict()

        for policy in self.policies:
            name = policy.__str__()
            cum_regret[name] = np.zeros((n_mc, n_sub))

        # run n_mc independent simulations
        for nExp in tqdm(range(n_mc)):
            if self.verbose:
                print('--------')
                print('Experiment number: ' + str(nExp))
                print('--------')

            for i, policy in enumerate(self.policies):
                # Reinitialize the policy
                time_init_pol = time.time()
                policy.re_init()
                self.mab.theta = self.theta
                name = policy.__str__()
                optimal_rewards = np.zeros(steps)
                rewards = np.zeros(steps)
                for t in range(steps):
                    if t in self.bp_dico.keys():
                        theta_new = self.bp_dico[t]
                        self.mab.theta = theta_new
                        if policy.omniscient:
                            policy.re_init()
                    if self.verbose:
                        print('time t=' + str(t))

                    available_arms = self.mab.get_arms(self.k, nExp)  # receiving K action vectors

                    _, instant_best_reward = self.mab.get_best_arm()
                    a_t = policy.select_arm(available_arms)  # number of the action
                    round_reward, a_t_features = self.mab.play(a_t)  # action_played is the feature vector
                    policy.update_state(a_t_features, round_reward)
                    expected_reward_round = self.mab.get_expected_rewards()[a_t]
                    optimal_rewards[t] = instant_best_reward
                    rewards[t] = expected_reward_round
                if self.verbose:
                    print('optimal_rewards: ', optimal_rewards)
                    print('rewards: ', rewards)
                    print('regret: ', cum_regret[name])
                cum_regret[name][nExp, :] = np.cumsum(optimal_rewards - rewards)[t_saved]

        print("-- Building data out of the experiments ---")

        for policy in self.policies:
            name = policy.__str__()
            cum_reg = cum_regret[name]  # Cumulative regret only on the t_saved points
            avg_regret[name] = np.mean(cum_reg, 0)
            q_regret[name] = np.percentile(cum_reg, q, 0)
            up_q_regret[name] = np.percentile(cum_reg, 100 - q, 0)

        print("--- Data built ---")
        return avg_regret, q_regret, up_q_regret