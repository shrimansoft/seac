import os

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces.box import Box
# from gymnasium.wrappers import Monitor
from gymnasium.wrappers import RecordVideo

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize as VecNormalize_
from wrappers import TimeLimit, RecordEpisodeStatistics


class MADummyVecEnv(DummyVecEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)
        agents = len(self.observation_space)
        # change this because we want >1 reward
        self.buf_rews = np.zeros((self.num_envs, agents), dtype=np.float32)

def make_env(env_id, seed, rank, time_limit, wrappers, monitor_dir):
    def _thunk():

        env = gym.make(env_id,render_mode="rgb_array")
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)

        if time_limit:
            env = TimeLimit(env, max_episode_steps=time_limit) # NOTE we can diractlly set the max_episode_steps in the make function

        for wrapper in wrappers:
            env = wrapper(env)
        
        if monitor_dir:
            # env = Monitor(env, monitor_dir, lambda ep: int(ep==0), force=True, uid=str(rank))
            trigger = lambda t: t % 10 == 0
            env = RecordVideo(env, video_folder=monitor_dir, episode_trigger=trigger, disable_logger=False)
            # env = RecordEpisodeStatistics(env)


        return env

    return _thunk


def make_vec_envs(
    env_name, seed, dummy_vecenv, parallel, time_limit, wrappers, device, monitor_dir=True
):
    envs = [
        make_env(env_name, seed, i, time_limit, wrappers,monitor_dir) for i in range(parallel)
    ]

    if dummy_vecenv or len(envs) == 1 or monitor_dir:
        print("Using DummyVecEnv")
        envs = MADummyVecEnv(envs)
    else:
        print("Using SubprocVecEnv")
        envs = SubprocVecEnv(envs, start_method="fork")

    envs = VecPyTorch(envs, device)
    return envs


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        return [torch.from_numpy(o).to(self.device) for o in obs]
        return obs

    def step_async(self, actions):
        actions = [a.squeeze().cpu().numpy() for a in actions]
        actions = list(zip(*actions))
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, rew, done, info = self.venv.step_wait()
        return (
            [torch.from_numpy(o).float().to(self.device) for o in obs],
            torch.from_numpy(rew).float().to(self.device),
            torch.from_numpy(done).float().to(self.device),
            info,
        )

