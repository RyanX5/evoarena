"""
sweep.py

Parameter sensitivity analysis — sweeps one parameter at a time while
holding the others at their defaults. Saves a comparison plot and raw
data for each parameter.

Run with:
    python experiments/sweep.py

Results saved to experiments/results/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from simulation.arena import Arena
from simulation.agent import Agent
from evolution.evolve import next_generation

GENERATIONS = 50
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# defaults
DEFAULT_MUTATION_RATE  = 0.1
DEFAULT_POPULATION     = 30
DEFAULT_ELITE_PCT      = 0.2


def run(population_size, mutation_rate, elite_pct, generations=GENERATIONS):
    """Run one simulation and return (best_history, avg_history)."""
    # patch config so existing code picks up the overrides
    config.POPULATION_SIZE = population_size
    config.MUTATION_RATE   = mutation_rate
    Arena._gen_count       = 0

    best_history = []
    avg_history  = []

    Agent._id_counter = 0
    agents = [Agent(0, 0) for _ in range(population_size)]

    for _ in range(generations):
        arena = Arena(width=config.ARENA_WIDTH, height=config.ARENA_HEIGHT)
        arena.reset(agents)
        Arena._gen_count += 1

        while arena.step_count < config.GEN_SIZE and len(arena.agents) > 1:
            arena.step()

        best = max(a.fitness() for a in arena.agents)
        avg  = sum(a.fitness() for a in arena.agents) / len(arena.agents)
        best_history.append(best)
        avg_history.append(avg)

        agents = next_generation_with_elite(arena.agents, population_size, mutation_rate, elite_pct)
        Agent._id_counter = 0

    return best_history, avg_history


def next_generation_with_elite(agents, population_size, mutation_rate, elite_pct):
    """Same as evolve.next_generation but with explicit params instead of config."""
    import random
    from simulation.neural_network import NeuralNetwork

    agents.sort(key=lambda a: a.fitness(), reverse=True)
    elite_count = max(1, int(len(agents) * elite_pct))
    elites = agents[:elite_count]

    new_agents = []
    while len(new_agents) < population_size:
        if len(elites) >= 2:
            parent_a = random.choice(elites)
            parent_b = random.choice([e for e in elites if e is not parent_a])
            wa = parent_a.brain.get_weights()
            wb = parent_b.brain.get_weights()
            mask = np.random.rand(len(wa)) > 0.5
            child_w = np.where(mask, wa, wb)
        else:
            child_w = elites[0].brain.get_weights().copy()

        child_w = child_w + np.random.normal(0, mutation_rate, size=child_w.shape)
        new_agents.append(Agent(x=0, y=0, brain=NeuralNetwork(weights=child_w)))

    return new_agents


def save_plot(results, param_name, values, filename):
    """Plot best fitness curves for each value of a parameter."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0f0f19")
    colors = ["#ffdc32", "#50c8c8", "#ff6b6b"]
    gens = list(range(1, GENERATIONS + 1))

    for i, val in enumerate(values):
        best_h, avg_h = results[val]
        label = f"{param_name}={val}"
        ax1.plot(gens, best_h, color=colors[i], linewidth=2, label=label)
        ax2.plot(gens, avg_h,  color=colors[i], linewidth=2, label=label)

    for ax, title in zip([ax1, ax2], ["Best Fitness", "Avg Fitness"]):
        ax.set_facecolor("#0f0f19")
        ax.set_title(title, color="white")
        ax.set_xlabel("Generation", color="white")
        ax.set_ylabel("Fitness", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#1a1a2e", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    fig.suptitle(f"Parameter Sweep: {param_name}", color="white", fontsize=13)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, facecolor="#0f0f19", dpi=120)
    plt.close(fig)
    print(f"  [plot saved] {path}")


def sweep_parameter(param_name, values, build_kwargs):
    """Run all values for one parameter and save results."""
    print(f"\nSweeping {param_name}: {values}")
    results = {}
    for val in values:
        kwargs = build_kwargs(val)
        print(f"  running {param_name}={val} ...", end=" ", flush=True)
        best_h, avg_h = run(**kwargs)
        results[val] = (best_h, avg_h)
        print(f"done  (final best={best_h[-1]:.1f})")

    # save raw data
    data = {str(v): {"best": results[v][0], "avg": results[v][1]} for v in values}
    json_path = os.path.join(RESULTS_DIR, f"sweep_{param_name}.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [data saved]  {json_path}")

    save_plot(results, param_name, values, f"sweep_{param_name}.png")


if __name__ == "__main__":
    # sweep mutation rate
    sweep_parameter(
        "mutation_rate",
        [0.05, 0.1, 0.2],
        lambda v: dict(population_size=DEFAULT_POPULATION, mutation_rate=v, elite_pct=DEFAULT_ELITE_PCT)
    )

    # sweep population size
    sweep_parameter(
        "population_size",
        [15, 30, 50],
        lambda v: dict(population_size=v, mutation_rate=DEFAULT_MUTATION_RATE, elite_pct=DEFAULT_ELITE_PCT)
    )

    # sweep elite %
    sweep_parameter(
        "elite_pct",
        [0.1, 0.2, 0.3],
        lambda v: dict(population_size=DEFAULT_POPULATION, mutation_rate=DEFAULT_MUTATION_RATE, elite_pct=v)
    )

    print("\nAll sweeps done. Results in experiments/results/")
