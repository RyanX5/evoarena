"""
main.py

Launches the arena with randomly spawned agents.
Each agent has their own NN (brain).
One every step (arena.step), it does a forward pass and recalculates vx and vy for each agent

Run with:
    python main.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from simulation.arena import Arena
from simulation.agent import Agent
from visualization.visualizer import Visualizer
from evolution.evolve import next_generation
from visualization.fitness_graph import FitnessGraph


def main():
    print("EvoArena — Report 1 Demo")
    print(f"  Population : {config.POPULATION_SIZE}")
    print(f"  Arena      : {config.ARENA_WIDTH}x{config.ARENA_HEIGHT}")
    print("  Press ESC or close window to quit.\n")

    
    best_fitness_history = []
    avg_fitness_history = []
    graph = FitnessGraph()

    Agent._id_counter = 0
    agents = [Agent(0, 0) for _ in range(config.POPULATION_SIZE)]

    arena = Arena(width=config.ARENA_WIDTH, height=config.ARENA_HEIGHT)
    arena.reset(agents)
    viz = Visualizer(arena=arena, fps=config.FPS)
    viz.init()

    

    while True:
        
        arena = Arena(width=config.ARENA_WIDTH, height=config.ARENA_HEIGHT)
        arena.reset(agents)
        viz.arena = arena
        Arena._gen_count += 1
        print(f"Generation: {Arena._gen_count}")

        while arena.step_count < config.GEN_SIZE and len(arena.agents) > 1:
            if viz.should_quit():
                viz.close()
                print("Done.")
                return
            arena.step()
            viz.draw()
        
        # computing fitness scores for graph
        best_fitness = max(a.fitness() for a in arena.agents)
        avg_fitness = sum(a.fitness() for a in arena.agents) / len(arena.agents)

        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        graph.update(best_fitness_history, avg_fitness_history)

        agents = next_generation(arena.agents)
        Agent._id_counter = 0

    


if __name__ == "__main__":
    main()
