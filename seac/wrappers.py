
import math
from collections import deque
from time import perf_counter

import gymnasium as  gym
import numpy as np
from gymnasium import ObservationWrapper, spaces
from gymnasium.wrappers import TimeLimit as GymTimeLimit
# from gymnasium.wrappers import Monitor as GymMonitor


class RecordEpisodeStatistics(gym.Wrapper):
    """ Multi-agent version of RecordEpisodeStatistics gym wrapper"""

    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.n_agents = len(env.observation_space)
        self.t0 = perf_counter()
        self.episode_reward = np.zeros(self.n_agents)
        self.episode_length = 0
        self.reward_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        observation,info = super().reset(**kwargs)
        self.episode_reward = np.array([0,0],dtype=np.float64)
        self.episode_length = 0
        self.t0 = perf_counter()
        info['episode_reward'] = self.episode_reward
        info['episode_length'] = self.episode_length
        info['episode_time'] = perf_counter() - self.t0

        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        done = [terminated, truncated]
        self.episode_reward += np.array(reward, dtype=np.float64)
        self.episode_length += 1
        if any(done):
            info["episode_reward"] = self.episode_reward
            for i, agent_reward in enumerate(self.episode_reward):
                info[f"agent{i}/episode_reward"] = agent_reward
            info["episode_length"] = self.episode_length
            info["episode_time"] = perf_counter() - self.t0

            self.reward_queue.append(self.episode_reward)
            self.length_queue.append(self.episode_length)
        return observation, reward, terminated,truncated, info


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple([
            spaces.flatten(obs_space, obs)
            for obs_space, obs in zip(self.env.observation_space, observation)
        ])


class SquashDones(gym.Wrapper):
    r"""
    NOTE : Not doing anyting. 
    Wrapper that squashes multiple dones to a single one using all(dones)"""

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = [terminated, truncated]
        return observation, reward, terminated, truncated, info


class GlobalizeReward(gym.RewardWrapper):

    def reward(self, reward):
        return self.n_agents * [sum(reward)]


class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env,max_episode_steps=max_episode_steps)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward,terminated, truncated, info = self.env.step(action)
        done = [truncated, terminated]
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not all(done)
            done = len(observation) * [True]
        return observation, reward, terminated,truncated, info

class ClearInfo(gym.Wrapper):
    def step(self, action):
        observation, reward, terminated,truncaed, info = self.env.step(action)
        return observation, reward,terminated,truncaed, {}


# class Monitor(GymMonitor):
#     def _after_step(self, observation, reward, done, info):
#         if not self.enabled: return done

#         if all(done) and self.env_semantics_autoreset:
#             # For envs with BlockingReset wrapping VNCEnv, this observation will be the first one of the new episode
#             self.reset_video_recorder()
#             self.episode_id += 1
#             self._flush()

#         # Record stats
#         self.stats_recorder.after_step(observation, sum(reward), all(done), info)
#         # Record video
#         self.video_recorder.capture_frame()

#         return done
