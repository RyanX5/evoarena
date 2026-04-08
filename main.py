"""
main.py

Launches the arena with randomly spawned agents.
Each agent has their own NN (brain).
Every step (arena.step), it does a forward pass and recalculates vx and vy for each agent

Run with:
    python main.py                          # normal visual mode
    python main.py --headless               # no window, runs fast
    python main.py --headless --generations 200
    python main.py --replay champions/gen_050.npy
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import config
from simulation.arena import Arena
from simulation.agent import Agent
from simulation.neural_network import NeuralNetwork
from visualization.fitness_graph import FitnessGraph
from evolution.evolve import next_generation


def save_champion(agent: Agent, gen: int):
    os.makedirs("champions", exist_ok=True)
    path = f"champions/gen_{gen:03d}.npy"
    np.save(path, agent.brain.get_weights())
    print(f"  [champion saved] {path}")


def replay(weights_path: str):
    """Load a saved champion and watch it fight fresh random agents."""
    from visualization.visualizer import Visualizer

    weights = np.load(weights_path)
    champ_brain = NeuralNetwork(weights=weights)

    print(f"Replaying champion from {weights_path}")
    print("  Press ESC or close window to quit.\n")

    arena = Arena(width=config.ARENA_WIDTH, height=config.ARENA_HEIGHT)
    viz = Visualizer(arena=arena, fps=config.FPS)
    viz.init()

    while True:
        Agent._id_counter = 0
        # champion is agent #1, rest are random
        champ = Agent(0, 0, brain=champ_brain.copy())
        opponents = [Agent(0, 0) for _ in range(config.POPULATION_SIZE - 1)]
        agents = [champ] + opponents

        arena = Arena(width=config.ARENA_WIDTH, height=config.ARENA_HEIGHT)
        arena.reset(agents)
        viz.arena = arena
        viz.pinned_champion_id = champ.id
        Arena._gen_count += 1
        print(f"Replay round {Arena._gen_count}")

        while arena.step_count < config.GEN_SIZE and len(arena.agents) > 1:
            if viz.should_quit():
                viz.close()
                return
            viz.draw()
            arena.step()

        print(f"  champ alive={champ.alive}  fitness={champ.fitness():.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="run without Pygame window")
    parser.add_argument("--generations", type=int, default=None, help="stop after N generations")
    parser.add_argument("--replay", type=str, default=None, help="path to a saved champion .npy file")
    args = parser.parse_args()

    if args.replay:
        replay(args.replay)
        return

    print("EvoArena")
    print(f"  Population : {config.POPULATION_SIZE}")
    print(f"  Arena      : {config.ARENA_WIDTH}x{config.ARENA_HEIGHT}")
    print(f"  Mode       : {'headless' if args.headless else 'visual'}")
    if not args.headless:
        print("  Press ESC or close window to quit.")
    print()

    best_fitness_history = []
    avg_fitness_history = []
    graph = FitnessGraph()

    viz = None
    if not args.headless:
        from visualization.visualizer import Visualizer
        dummy_arena = Arena(width=config.ARENA_WIDTH, height=config.ARENA_HEIGHT)
        viz = Visualizer(arena=dummy_arena, fps=config.FPS)
        viz.init()

    Agent._id_counter = 0
    agents = [Agent(0, 0) for _ in range(config.POPULATION_SIZE)]

    while True:
        arena = Arena(width=config.ARENA_WIDTH, height=config.ARENA_HEIGHT)
        arena.reset(agents)
        Arena._gen_count += 1

        if viz:
            viz.arena = arena

        print(f"Generation: {Arena._gen_count}")

        while arena.step_count < config.GEN_SIZE and len(arena.agents) > 1:
            if viz:
                if viz.should_quit():
                    viz.close()
                    print("Done.")
                    return
                viz.draw()
            arena.step()

        best_agent = max(arena.agents, key=lambda a: a.fitness())
        best_fitness = best_agent.fitness()
        avg_fitness = sum(a.fitness() for a in arena.agents) / len(arena.agents)
        print(f"  best={best_fitness:.1f}  avg={avg_fitness:.1f}  survivors={len(arena.agents)}")

        save_champion(best_agent, Arena._gen_count)

        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        graph.update(best_fitness_history, avg_fitness_history)

        agents = next_generation(arena.agents)
        Agent._id_counter = 0

        if args.generations and Arena._gen_count >= args.generations:
            if viz:
                viz.close()
            print("Done.")
            return


if __name__ == "__main__":
    main()
