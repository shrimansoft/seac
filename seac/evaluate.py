import torch
import rware
import lbforaging
import gymnasium as gym

from a2c import A2C
from wrappers import RecordEpisodeStatistics, TimeLimit

path = "results/trained_models/0/u500"
env_name = "rware-tiny-2ag-v2"
time_limit = 500 # 25 for LBF

EPISODES = 5

env = gym.make(env_name)
agents = [
    A2C(i, osp, asp, 0.1, 0.1, False, 1, 1, "cpu")
    for i, (osp, asp) in enumerate(zip(env.observation_space, env.action_space))
]
for agent in agents:
    agent.restore(path + f"/agent{agent.agent_id}")

for ep in range(EPISODES):
    env = gym.make(env_name,render_mode="rgb_array",max_episode_steps=100)
    # env = Monitor(env, f"seac_rware-small-4ag_eval/video_ep{ep+1}", mode="evaluation")
    env = TimeLimit(env, time_limit)
    env = RecordEpisodeStatistics(env)

    obs, info = env.reset()  # Ensure reset returns a tuple
    done = False

    while not done:
        obs = [torch.from_numpy(o) for o in obs]
        _, actions, _ , _ = zip(*[agent.model.act(obs[agent.agent_id], None, None) for agent in agents])
        actions = [a.item() for a in actions]
        obs, _, terminated,truncated, info = env.step(actions)
        done = terminated or truncated
        env.render()
    env.close()
    obs, info = env.reset()  # Ensure reset returns a tuple
    print("--- Episode Finished ---")
    print(f"Episode rewards: {sum(info['episode_reward'])}")
    print(info)
    print(" --- ")
