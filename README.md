# overcooked-ppo-dqn
Comparing PPO and DQN robustness to teammate behavior changes in Overcooked-AI

## quick start
```
uv sync
uv run python tests/setup_check.py
uv run python src/baseline/run_greedy.py 
```

setup_check 会检查需要的环境
run_greedy 是基线，既两个greedy model的agent pair
