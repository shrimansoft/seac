import glob
import logging
import os
import shutil
import time
from collections import deque
from os import path
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from sacred import Experiment
from sacred.observers import (  # noqa
    FileStorageObserver, MongoObserver, QueuedMongoObserver, QueueObserver,
)
from torch.utils.tensorboard.writer import SummaryWriter

import utils
from a2c import A2C, algorithm
from envs import make_vec_envs
from wrappers import RecordEpisodeStatistics, SquashDones
from model import Policy

import rware as robotic_warehouse  # noqa
import lbforaging  # noqa

ex = Experiment(ingredients=[algorithm], name="A2C", interactive=True)
# ex = Experiment(ingredients=[algorithm])
ex.captured_out_filter = lambda captured_output: "Output capturing turned off."  # TODO: What does this do?
ex.observers.append(FileStorageObserver("./results/sacred"))

logging.basicConfig(
    level=logging.INFO,
    format=
    "(%(process)d) [%(levelname).1s] - (%(asctime)s) - %(name)s >> %(message)s",
    datefmt="%m/%d %H:%M:%S",
)


@ex.config
def config():
    env_name = None
    time_limit = None
    wrappers = (
        RecordEpisodeStatistics,
        SquashDones,
    )
    dummy_vecenv = False

    num_env_steps = 100e6

    eval_dir = "./results/video/{id}"
    loss_dir = "./results/loss/{id}"
    save_dir = "./results/trained_models/{id}"

    log_interval = 2000
    save_interval = int(1e6)
    eval_interval = int(1e6)
    episodes_per_eval = 8


for conf in glob.glob("configs/*.yaml"):
    name = f"{Path(conf).stem}"
    ex.add_named_config(name, conf)


def _squash_info(info):
    info = [i for i in info if i]
    new_info = {}
    keys = set([k for i in info for k in i.keys()])
    keys.discard("TimeLimit.truncated")
    for key in keys:
        mean = np.mean([np.array(d[key]).sum() for d in info if key in d])
        new_info[key] = mean
    return new_info


@ex.capture
def evaluate(
    agents,
    monitor_dir,
    episodes_per_eval,
    env_name,
    seed,
    wrappers,
    dummy_vecenv,
    time_limit,
    algorithm,
    _log,
):
    device = algorithm["device"]

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
        torch.zeros(episodes_per_eval,
                    agent.model.recurrent_hidden_state_size,
                    device=device) for agent in agents
    ]
    masks = torch.zeros(episodes_per_eval, 1, device=device)

    all_infos = []

    while len(all_infos) < episodes_per_eval:
        with torch.no_grad():
            _, n_action, _, n_recurrent_hidden_states = zip(*[
                agent.model.act(n_obs[agent.agent_id], recurrent_hidden_states,
                                masks)
                for agent, recurrent_hidden_states in zip(
                    agents, n_recurrent_hidden_states)
            ])

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


@ex.main
def main(
    _run,
    _log,
    num_env_steps,
    env_name,
    seed,
    algorithm,
    dummy_vecenv,
    time_limit,
    wrappers,
    save_dir,
    eval_dir,
    loss_dir,
    log_interval,
    save_interval,
    eval_interval,
):
    _log.info("Starting training")

    if loss_dir:
        loss_dir = path.expanduser(loss_dir.format(id=str(_run._id)))
        print(loss_dir)
        utils.cleanup_log_dir(loss_dir)
        writer = SummaryWriter(loss_dir)
    else:
        writer = None

    eval_dir = path.expanduser(eval_dir.format(id=str(_run._id)))
    save_dir = path.expanduser(save_dir.format(id=str(_run._id)))

    utils.cleanup_log_dir(eval_dir)
    utils.cleanup_log_dir(save_dir)

    torch.set_num_threads(1)

    print("env_name: ", env_name)
    envs = gym.make_vec(env_name,num_envs=4,vectorization_mode="sync",
                        wrappers=(lambda x: gym.wrappers.TimeLimit(x,max_episode_steps=1000),))

    envs2 = make_vec_envs(
        env_name,
        seed,
        dummy_vecenv,
        algorithm["num_processes"],
        time_limit,
        wrappers,
        algorithm["device"],
    )

    print("----------------------***********----------------------")
    print("envs", envs)
    print("envs2", envs2)
    print("envs.observation_space", envs.observation_space)
    print("envs2.observation_space", envs2.observation_space[0].shape)
    print("envs.action_space", envs.action_space[0].nvec)
    print("envs2.action_space", envs2.action_space[0].n)
    print("device", algorithm["device"])

    agents = [
        A2C(i, osp, asp) for i, (osp, asp) in enumerate(zip(envs.observation_space, envs.action_space))
    ]
    print("agents", agents[0])

    obs = envs.reset()
    # print("obs", obs)

    for i in range(len(obs)):
        agents[i].storage.obs[0].copy_(obs[i])
        agents[i].storage.to(algorithm["device"])

    start = time.time()
    num_updates = (int(num_env_steps) // algorithm["num_steps"] //
                   algorithm["num_processes"])

    all_infos = deque(maxlen=10)

    # for j in range(1, num_updates + 1):
    for j in range(1, 10):

        for step in range(algorithm["num_steps"]):
            # Sample actions
            with torch.no_grad():
                n_value, n_action, n_action_log_prob, n_recurrent_hidden_states = zip(
                    *[
                        agent.model.act(
                            agent.storage.obs[step],
                            agent.storage.recurrent_hidden_states[step],
                            agent.storage.masks[step],
                        ) for agent in agents
                    ])
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(n_action)
            # envs.envs[0].render()

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])

            bad_masks = torch.FloatTensor(
                [[0.0] if info.get("TimeLimit.truncated", False) else [1.0]
                 for info in infos])
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

        if j % log_interval == 0 and len(all_infos) > 1:
            squashed = _squash_info(all_infos)

            total_num_steps = ((j + 1) * algorithm["num_processes"] *
                               algorithm["num_steps"])
            end = time.time()
            _log.info(
                f"Updates {j}, num timesteps {total_num_steps}, FPS {int(total_num_steps / (end - start))}"
            )
            _log.info("hello world")
            # _log.info(
            #     f"Last {len(all_infos)} training episodes mean reward {squashed['episode_reward'].sum():.3f}"
            # )

            for k, v in squashed.items():
                _run.log_scalar(k, v, j)
            all_infos.clear()

        if save_interval is not None and (j > 0 and j % save_interval == 0
                                          or j == num_updates):
            cur_save_dir = path.join(save_dir, f"u{j}")
            for agent in agents:
                save_at = path.join(cur_save_dir, f"agent{agent.agent_id}")
                os.makedirs(save_at, exist_ok=True)
                agent.save(save_at)
            archive_name = shutil.make_archive(cur_save_dir, "xztar", save_dir,
                                               f"u{j}")
            shutil.rmtree(cur_save_dir)
            _run.add_artifact(archive_name)

        if eval_interval is not None and (j > 0 and j % eval_interval == 0
                                          or j == num_updates):
            evaluate(
                agents,
                os.path.join(eval_dir, f"u{j}"),
            )
            videos = glob.glob(os.path.join(eval_dir, f"u{j}") + "/*.mp4")
            for i, v in enumerate(videos):
                _run.add_artifact(v, f"u{j}.{i}.mp4")
    envs.close()


ex.run(config_updates={"env_name": "rware-tiny-2ag-v2", "time_limit": 1})
