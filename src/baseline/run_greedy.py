import numpy as np
from overcooked_ai_py.agents.agent import GreedyHumanModel, AgentPair
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

def main():
    np.random.seed(0)
    
    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    env = OvercookedEnv.from_mdp(mdp, horizon=400, info_level=0)
    
    print("Preparing planner...", flush=True)
    mlam = env.mlam
    
    pair = AgentPair(GreedyHumanModel(mlam), GreedyHumanModel(mlam))
    
    scores = []
    for episode in range(5):
        env.reset()
        
        pair.reset()
        pair.set_mdp(env.mdp)
        trajectory, steps, sparse_reward, shaped_reward = env.run_agents(
            pair,
            display=False,
        )
        scores.append(sparse_reward)
        
        print(
            f"Episode {episode + 1}: "
            f"steps={steps}, "
            f"sparse_reward={sparse_reward}, "
            f"shaped_reward={shaped_reward}"
        )
    print(f"\nMean sparse reward: {np.mean(scores):.2f}")
    
if __name__ == "__main__":
    main()
    