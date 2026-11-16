"""Run after `uv sync`: uv run python tests/setup_check.py.

Checks the project interpreter, dependency imports, and a tiny Overcooked
rollout. Does not train models, render graphics, or compute planner caches.
"""

import importlib
import platform
import sys
from importlib.metadata import version
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check_python():
    expected = (PROJECT_ROOT / ".python-version").read_text().strip()
    actual = platform.python_version()
    if actual != expected:
        raise RuntimeError(
            f"Expected Python {expected} from .python-version, found {actual}. "
            f"Run: uv sync --python {expected}"
        )
    print(f"[PASS] Python {actual}")


def check_virtualenv():
    expected = (PROJECT_ROOT / ".venv").resolve()
    if sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != expected:
        raise RuntimeError(
            "Not using this project's .venv. "
            "Run from the project root: uv run python tests/setup_check.py"
        )
    print("[PASS] Project virtual environment")


def check_dependency(module_name, distribution_name):
    importlib.import_module(module_name)
    print(f"[PASS] {distribution_name} {version(distribution_name)}")


def check_overcooked():
    import numpy as np

    from overcooked_ai_py.agents.agent import AgentPair, GreedyHumanModel
    from overcooked_ai_py.mdp.actions import Action
    from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    from stable_baselines3 import DQN, PPO

    # Resolve the symbols used by our baseline and future training scripts.
    if not all(callable(cls) for cls in (AgentPair, GreedyHumanModel, PPO, DQN)):
        raise RuntimeError("Required agent/algorithm classes are unavailable")

    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    env = OvercookedEnv.from_mdp(mdp, horizon=2, info_level=0)

    # Test MDP transitions directly: env.step lazily computes a motion planner.
    # This setup check deliberately avoids creating cache files in site-packages.
    state = env.state
    encoded = mdp.lossless_state_encoding(state)
    if len(encoded) != 2 or not all(np.isfinite(obs).all() for obs in encoded):
        raise RuntimeError("Invalid state encoding")

    for _ in range(2):
        state, transition_info = mdp.get_state_transition(
            state, (Action.STAY, Action.STAY)
        )
        if "sparse_reward_by_agent" not in transition_info:
            raise RuntimeError("Missing reward information")

    env.state = state
    if state.timestep != 2 or not env.is_done():
        raise RuntimeError("Episode did not end at the configured horizon")
    env.reset(regen_mdp=False)
    if env.state.timestep != 0 or env.is_done():
        raise RuntimeError("Environment reset failed")
    print("[PASS] cramped_room: encoding, 2-step transition, horizon, reset")


def main():
    print("Checking project setup...\n", flush=True)
    failures = []

    def run_check(label, check):
        try:
            check()
        except Exception as exc:
            failures.append(label)
            print(f"[FAIL] {label}: {type(exc).__name__}: {exc}", flush=True)

    run_check("Python version", check_python)
    run_check("Virtual environment", check_virtualenv)

    dependencies = [
        ("numpy", "numpy"),
        ("gymnasium", "gymnasium"),
        ("torch", "torch"),
        ("stable_baselines3", "stable-baselines3"),
        ("overcooked_ai_py", "overcooked-ai"),
    ]
    for module_name, distribution_name in dependencies:
        run_check(
            distribution_name,
            lambda m=module_name, d=distribution_name: check_dependency(m, d),
        )

    if not failures:
        run_check("Overcooked smoke test", check_overcooked)
    else:
        print("[SKIP] Overcooked smoke test: fix the checks above first")

    if failures:
        print(f"\nSetup check FAILED ({len(failures)} failed checks).")
        print("Run `uv sync --locked` in the project root, then retry.")
        return 1

    print("\nSetup check PASSED. Dependencies and basic environment are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
