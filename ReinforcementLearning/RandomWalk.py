# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:02:23 2020
@author: Sila
"""

# Reinforcement Learning.
# The RandomWalk example 6.2 from RL by Barto and Sutton.

import numpy as np
import matplotlib.pyplot as plt

# 0 is the left terminal state
# 10 is the right terminal state
# 1 ... 10 represents A ... J
# See drawing in exercises.

VALUES = np.zeros(11)
VALUES[1:10] = 0.5
# For convenience, we assume all rewards are 0
# and the left terminal state has value 0, the right terminal state has value 1
# This trick has been used in Gambler's Problem
VALUES[10] = 1

# set up true state values
TRUE_VALUE = np.zeros(11)
TRUE_VALUE[1:10] = np.arange(1, 10) / 10.0
TRUE_VALUE[10] = 1

ACTION_LEFT = 0
ACTION_RIGHT = 1

# @values: current states value
# @alpha: step size
def temporal_difference(values, alpha=0.1):
    state = 5
    trajectory = [state]
    rewards = [0]
    while True:
        old_state = state
        if np.random.binomial(1, 0.5) == ACTION_LEFT:
            state -= 1
        else:
            state += 1
        # Assume all rewards are 0
        reward = 0
        trajectory.append(state)
        # TD update
        values[old_state] += alpha * (reward + values[state] - values[old_state])
        if state == 10 or state == 0:
            break
        rewards.append(reward)
    return trajectory, rewards

# @values: current states value
# @alpha: step size
def monte_carlo(values, alpha=0.1):
    state = 5
    trajectory = [5]

    # if end up with left terminal state, all returns are 0
    # if end up with right terminal state, all returns are 1
    while True:
        if np.random.binomial(1, 0.5) == ACTION_LEFT:
            state -= 1
        else:
            state += 1
        trajectory.append(state)
        if state == 10:
            returns = 1.0
            break
        elif state == 0:
            returns = 0.0
            break

    for state_ in trajectory[:-1]:
        # MC update
        values[state_] += alpha * (returns - values[state_])

    return trajectory, [returns] * (len(trajectory) - 1)

# Example 6.2 in RL by Barto and Sutton. TD update.
def compute_td_state_value():
    episodes = [0, 1, 10, 50, 200]
    current_values = np.copy(VALUES)
    plt.figure(1)
    for i in range(episodes[-1] + 1):
        if i in episodes:
            plt.plot(current_values, label=str(i) + ' episodes')
        temporal_difference(current_values)
    plt.plot(TRUE_VALUE, label='true values')
    plt.xlabel('state')
    plt.ylabel('estimated value')
    plt.legend()

# Example 6.2 in RL by Barto and Sutton. MC update.
def compute_mc_state_value():
    episodes = [0, 1, 10, 50, 200]
    current_values = np.copy(VALUES)
    plt.figure(1)
    for i in range(episodes[-1] + 1):
        if i in episodes:
            plt.plot(current_values, label=str(i) + ' episodes')
        monte_carlo(current_values)
    plt.plot(TRUE_VALUE, label='true values')
    plt.xlabel('state')
    plt.ylabel('estimated value')
    plt.legend()

# Example 6.2 in RL by Barto and Sutton. TD and MC update.
def rl_example_6_2():
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    compute_td_state_value()
    plt.subplot(2, 1, 2)
    compute_mc_state_value()
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    rl_example_6_2()
