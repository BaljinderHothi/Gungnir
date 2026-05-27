# Gungir
Gungnir is a distributed async RL system built around stateless rollout workers, a centralized inference server, and a decoupled learner. The environment is a plugin, swap in a MuJoCo locomotion task, a manipulation env, or an LLM generating tokens against a reward model. Same architecture, same algorithms, different domain.

Built to run on a single consumer GPU while being architecturally honest and sound

## Algorithms
- PPO with GAE advantage estimation and KL penalty
- GRPO (Group Relative Policy Optimization)
- V-trace for off-policy importance sampling correction

## Environments
- MuJoCo continuous control (locomotion, loco-manipulation)
- PushT manipulation env
- LLM env — token generation scored against a reward model

## Quickstart
git clone https://github.com/BaljinderHothi/gungnir
cd gungnir
pip install -r requirements.txt

### robotics
python driver.py --config configs/ppo_mujoco.yaml

### LLM post-training
python driver.py --config configs/ppo_llm.yaml

# Status
MuJoCo env + PPO — in progress
PushT + V-trace — planned
LLM env + GRPO — planned
