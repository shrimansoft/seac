import glob
import logging
import os
import shutil
import time
from collections import deque
from os import path
from pathlib import Path
import argparse

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

import utils
from a2c import A2C
from envs import make_vec_envs
from wrappers import RecordEpisodeStatistics, SquashDones
from model import Policy

import rware as robotic_warehouse # noqa
import lbforaging # noqa

logging.basicConfig(
    level=logging.INFO,
    format="(%(process)d) [%(levelname).1s] - (%(asctime)s) - %(name)s >> %(message)s",
    datefmt="%m/%d %H:%M:%S",
)

def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--env_name", type=str, required=True, help="Environment name")
    parser.add_argument("--time_limit", type=int, default=None, help="Time limit")
    parser.add_argument("--num_env_steps", type=float, default=100e6, help="Number of environment steps")
    parser.add_argument("--eval_dir", type=str, default="./results/video/{id}", help="Evaluation directory")
    parser.add_argument("--loss_dir", type=str, default="./results/loss/{id}", help="Loss directory")
    parser.add_argument("--save_dir", type=str, default="./results/trained_models/{id}", help="Save directory")
    parser.add_argument("--log_interval", type=int, default=2000, help="Log interval")
    parser.add_argument("--save_interval", type=int, default=int(1e6), help="Save interval")
    parser.add_argument("--eval_interval", type=int, default=int(1e6), help="Evaluation interval")
    parser.add_argument("--episodes_per_eval", type=int, default=8, help="Episodes per evaluation")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--adam_eps", type=float, default=0.001, help="Adam epsilon")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--use_gae", action="store_true", help="Use GAE")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--value_loss_coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--use_proper_time_limits", action="store_true", help="Use proper time limits")
    parser.add_argument("--recurrent_policy", action="store_true", help="Use recurrent policy")
    parser.add_argument("--use_linear_lr_decay", action="store_true", help="Use linear learning rate decay")
    parser.add_argument("--seac_coef", type=float, default=1.0, help="SEAC coefficient")
    parser.add_argument("--num_processes", type=int, default=4, help="Number of processes")
    parser.add_argument("--num_steps", type=int, default=5, help="Number of steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()

def _squash_info(info):
    info = [i for i in info if i]
    new_info = {}
    keys = set([k for i in info for k in i.keys()])
    keys.discard("TimeLimit.truncated")
    for key in keys:
        mean = np.mean([np.array(d[key]).sum() for d in info if key in d])
        new_info[key] = mean
    return new_info

def evaluate(
    agents,
    monitor_dir,
    episodes_per_eval,
    env_name,
    seed,
    wrappers,
    dummy_vecenv,
    time_limit,
    args,
    _log,
):
    device = args.device

    eval_envs = make_vec_envs(
        env_name,
        seed,
        dummy_vecenv,
        episodes_per_eval,
        time_limit,
        wrappers,
        device,
        monitor_dir=monitor_dir,
    )

    n_obs = eval_envs.reset()
    n_recurrent_hidden_states = [
        torch.zeros(
            episodes_per_eval, agent.model.recurrent_hidden_state_size, device=device
        )
        for agent in agents
    ]
    masks = torch.zeros(episodes_per_eval, 1, device=device)

    all_infos = []

    while len(all_infos) < episodes_per_eval:
        with torch.no_grad():
            _, n_action, _, n_recurrent_hidden_states = zip(
                *[
                    agent.model.act(
                        n_obs[agent.agent_id], recurrent_hidden_states, masks
                    )
                    for agent, recurrent_hidden_states in zip(
                        agents, n_recurrent_hidden_states
                    )
                ]
            )

        # Obser reward and next obs
        n_obs, _, done, infos = eval_envs.step(n_action)

        n_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device,
        )
        all_infos.extend([i for i in infos if i])

    eval_envs.close()
    info = _squash_info(all_infos)
    _log.info(
        f"Evaluation using {len(all_infos)} episodes: mean reward {info['episode_reward']:.5f}\n"
    )

def main():
    args = parse_args()

    if args.loss_dir:
        loss_dir = path.expanduser(args.loss_dir.format(id=str(args.seed)))
        print(loss_dir)
        utils.cleanup_log_dir(loss_dir)
        writer = SummaryWriter(loss_dir)
    else:
        writer = None

    eval_dir = path.expanduser(args.eval_dir.format(id=str(args.seed)))
    save_dir = path.expanduser(args.save_dir.format(id=str(args.seed)))

    utils.cleanup_log_dir(eval_dir)
    utils.cleanup_log_dir(save_dir)

    torch.set_num_threads(1)

    print("env_name", args.env_name)
    envs = make_vec_envs(
        args.env_name,
        args.seed,
        False,
        args.num_processes,
        args.time_limit,
        (RecordEpisodeStatistics, SquashDones),
        args.device,
        monitor_dir=True
    )

    print("Sucessfully created envs")

    agents = [
        A2C(i, osp, asp)
        for i, (osp, asp) in enumerate(zip(envs.observation_space, envs.action_space))
    ]

    obs = envs.reset()
    print("Sucessfully reset envs", envs)

    for i in range(len(obs)):
        agents[i].storage.obs[0].copy_(obs[i])
        agents[i].storage.to(args.device)

    start = time.time()
    num_updates = (
        int(args.num_env_steps) // args.num_steps // args.num_processes)

    all_infos = deque(maxlen=10)

    for j in range(1, num_updates + 1):

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                n_value, n_action, n_action_log_prob, n_recurrent_hidden_states = zip(
                    *[
                        agent.model.act(
                            agent.storage.obs[step],
                            agent.storage.recurrent_hidden_states[step],
                            agent.storage.masks[step],
                        )
                        for agent in agents
                    ]
                )
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(n_action)
            # envs.envs[0].render()

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            bad_masks = torch.FloatTensor(
                [
                    [0.0] if info.get("TimeLimit.truncated", False) else [1.0]
                    for info in infos
                ]
            )
            for i in range(len(agents)):
                agents[i].storage.insert(
                    obs[i],
                    n_recurrent_hidden_states[i],
                    n_action[i],
                    n_action_log_prob[i],
                    n_value[i],
                    reward[:, i].unsqueeze(1),
                    masks,
                    bad_masks,
                )

            for info in infos:
                if info:
                    all_infos.append(info)

        # value_loss, action_loss, dist_entropy = agent.update(rollouts)
        for agent in agents:
            agent.compute_returns()

        for agent in agents:
            loss = agent.update([a.storage for a in agents])
            for k, v in loss.items():
                if writer:
                    writer.add_scalar(f"agent{agent.agent_id}/{k}", v, j)

        for agent in agents:
            agent.storage.after_update()

        if j % args.log_interval == 0 and len(all_infos) > 1:
            squashed = _squash_info(all_infos)

            total_num_steps = (
                (j + 1) * args.num_processes * args.num_steps 
            )
            end = time.time()
            logging.info(
                f"Updates {j}, num timesteps {total_num_steps}, FPS {int(total_num_steps / (end - start))}"
            )
            # logging.info(
            #     f"Last {len(all_infos)} training episodes mean reward {squashed['episode_reward'].sum():.3f}"
            # )

            for k, v in squashed.items():
                logging.info(f"{k}: {v}")
            all_infos.clear()

        if args.save_interval is not None and (
            j > 0 and j % args.save_interval == 0 or j == num_updates
        ):
            cur_save_dir = path.join(save_dir, f"u{j}")
            for agent in agents:
                save_at = path.join(cur_save_dir, f"agent{agent.agent_id}")
                os.makedirs(save_at, exist_ok=True)
                agent.save(save_at)
            archive_name = shutil.make_archive(cur_save_dir, "xztar", save_dir, f"u{j}")
            shutil.rmtree(cur_save_dir)

        if args.eval_interval is not None and (
            j > 0 and j % args.eval_interval == 0 or j == num_updates
        ):
            evaluate(
                agents, os.path.join(eval_dir, f"u{j}"),
                args.episodes_per_eval,
                args.env_name,
                args.seed,
                (RecordEpisodeStatistics, SquashDones),
                False,
                args.time_limit,
                args,
                logging,
            )
            videos = glob.glob(os.path.join(eval_dir, f"u{j}") + "/*.mp4")
            for i, v in enumerate(videos):
                logging.info(f"Video {i}: {v}")
    envs.close()

if __name__ == "__main__":
    main()
