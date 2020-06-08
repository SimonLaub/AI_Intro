# -*- coding: utf-8 -*-
"""
Taxi simulator with q-learning.
Gym Taxi-v3
Created 02-07-2018
"""

from collections import defaultdict
from collections import deque
import sys
import math
import gym
import numpy as np

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.
        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.episodes = 1

    def select_action(self, state):
        """ Given the state, select an action.
        Params
        ======
        - state: the current state of the environment
        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # Select next action with greedy policy
        #epsilon = 0.017
        #epsilon = 1 / self.episodes
        epsilon = min(1 / self.episodes, 0.0001)
        probs = epsilon * np.ones(self.nA) / self.nA
        probs[np.argmax(self.Q[state])] = 1 - epsilon + epsilon / self.nA
        return np.random.choice(range(self.nA), p=probs)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # Actually we are already at next state applying the policy (greedy policy)
        # We exploit all the information from this step to update Q
        gamma = 0.95
        alpha = 0.1

        # Update Q consider next action as best action (Q-learning), instead of greedy policy
        Q_next = 0
        if not done:
            best_action = np.argmax(self.Q[next_state])
            Q_next = self.Q[next_state][best_action]
        else:
            self.episodes += 1
            Q_next = 0  # when next state is terminal, Q[state_terminal][.] = 0, no next_action available

        # Update Q. r_t+1 is the reward after action (a) before taking next_action a_t+1
        self.Q[state][action] = self.Q[state][action] + alpha * (reward + gamma * Q_next - self.Q[state][action])

def interact(env, agent, num_episodes=10000, window=35):
    """ Monitor agent's performance.

    Params
    ======
    - env: instance of OpenAI Gym's Taxi-v3 environment
    - agent: instance of class Agent
    - num_episodes: number of episodes of agent-environment interaction
    - window: number of episodes to consider when calculating average rewards
    Returns
    =======
    - avg_rewards: deque containing average rewards
    - best_avg_reward: largest value in the avg_rewards deque
    """
    # initialize average rewards
    avg_rewards = deque(maxlen=num_episodes)
    # initialize best average reward
    best_avg_reward = -math.inf
    # initialize monitor for most recent rewards
    samp_rewards = deque(maxlen=window)
    # for each episode
    for i_episode in range(1, num_episodes + 1):
        # begin the episode
        state = env.reset()
        # initialize the sampled reward
        samp_reward = 0
        while True:
            # agent selects an action
            action = agent.select_action(state)
            # agent performs the selected action
            next_state, reward, done, _ = env.step(action)
            # agent performs internal updates based on sampled experience
            agent.step(state, action, reward, next_state, done)
            # update the sampled reward
            samp_reward += reward
            # update the state (s <- s') to next time step
            state = next_state
            if done:
                # save final sampled reward
                samp_rewards.append(samp_reward)
                break
        if (i_episode >= 100):
            # get average reward from last 100 episodes
            avg_reward = np.mean(samp_rewards)
            # append to deque
            avg_rewards.append(avg_reward)
            # update best average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
        # monitor progress
        print("\rEpisode {}/{} || Best average reward {}".format(i_episode, num_episodes, best_avg_reward), end="")
        sys.stdout.flush()
        # check if task is solved (according to OpenAI Gym)
        if best_avg_reward >= 9.7:
            print('\nEnvironment solved in {} episodes.'.format(i_episode), end="")
            break
        if i_episode == num_episodes: print('\n')
    return avg_rewards, best_avg_reward

env = gym.make('Taxi-v3')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)  # Start the simulation
