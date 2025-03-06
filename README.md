# Shared Experience Actor Critic

This repository is the official implementation of [Shared Experience Actor Critic](https://arxiv.org/abs/2006.07169). 

## Requirements

For the experiments in LBF and RWARE, please install from:
- [Level Based Foraging Official Repo](https://github.com/uoe-agents/lb-foraging)
- [Multi-Robot Warehouse Official Repo](https://github.com/uoe-agents/robotic-warehouse)

Also requires, PyTorch 1.6+

## Setup

We use Poetry for dependency management. To install Poetry, follow the instructions [here](https://python-poetry.org/docs/#installation).

To set up the environment, run the following commands:

```bash
# Clone the repository
git clone https://github.com/your-repo/seac.git
cd seac

# Install dependencies
poetry install

# Activate the virtual environment
source "$( poetry env list --full-path | grep Activated | cut -d' ' -f1 )/bin/activate"
```

the above command is taken from [stackoverflow](https://stackoverflow.com/q/60580332)


## Training - SEAC

To train the agents, navigate to the `seac` directory and run the training script with the desired configuration:

```bash
cd seac
python train.py --env_name=rware-tiny-2ag-v2 --time_limit=500
```

You can customize the training by passing different arguments. For example:

```bash
python train.py --env_name=Foraging-15x15-3p-4f-v0 --time_limit=25
```

Here are some valid environment configurations:
- `--env_name=Foraging-15x15-3p-4f-v0 --time_limit=25`
- `--env_name=Foraging-12x12-2p-1f-v0 --time_limit=25`
- `--env_name=rware-tiny-2ag-v1 --time_limit=500`
- `--env_name=rware-tiny-4ag-v1 --time_limit=500`

## Training - SEQL

To train the agents in SEQL, navigate to the `seql` directory and run the training script. Possible options are:

```bash
cd seql
python lbf_train.py --env Foraging-12x12-2p-1f-v0
python lbf_train.py --env Foraging-15x15-3p-4f-v0
python rware_train.py --env "rware-tiny-2ag-v1"
python rware_train.py --env "rware-tiny-4ag-v1"
```

## Evaluation/Visualization - SEAC

To load and render the pretrained models in SEAC, run in the `seac` directory:

```bash
python evaluate.py
```


## Tools
### To remove the not good results in bulk 

```shell
find . -type d -regex './\(loss\|trained_models\|video\)/\(9\|20\|19\|18\|17\|10\)' -exec rm -r {} +
```

## Citation
```
@inproceedings{christianos2020shared,
  title={Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning},
  author={Christianos, Filippos and Sch{\"a}fer, Lukas and Albrecht, Stefano V},
  booktitle = {Advances in Neural Information Processing Systems},
  year={2020}
}
```
