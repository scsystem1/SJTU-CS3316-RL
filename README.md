# Dueling DDQN and PPO in Classic RL Environments

---

## Implementation & Analysis Details

All implementation details, model architectures, hyperparameters, and experiment results are described in **RL_PJ.pdf**.

---

## Requirements

- Python 3.x
- Gymnasium
- gym
- ALE Atari environment  
  (for `ALE/Pong-v5`, requires `ale-py`)
- MuJoCo environments  
  (for `Hopper-v4`, `Ant-v4`, `HalfCheetah-v4`; requires `mujoco` and `mujoco-py` bindings)
- PyTorch
- numpy
- tqdm

## Run
To run the Dueling DDQN agent on the Pong environment, use:

```bash
python run.py --env_name ALE/Pong-v5
```

To run the PPO agent on the Hopper environment, use:

```bash
python run.py --env_name Hopper-v4
```
To run the PPO agent on the Ant environment, use:

```bash
python run.py --env_name Ant-v4
```
To run the PPO agent on the HalfCheetah environment, use:

```bash
python run.py --env_name HalfCheetah-v4
```