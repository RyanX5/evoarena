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


def main():
    print("EvoArena — Report 1 Demo")
    print(f"  Population : {config.POPULATION_SIZE}")
    print(f"  Arena      : {config.ARENA_WIDTH}x{config.ARENA_HEIGHT}")
    print("  Press ESC or close window to quit.\n")

    arena = Arena(width=config.ARENA_WIDTH, height=config.ARENA_HEIGHT)

    Agent._id_counter = 0
    agents = [Agent(0, 0) for _ in range(config.POPULATION_SIZE)]
    arena.reset(agents)

    viz = Visualizer(arena, fps=config.FPS)
    viz.init()

    while True:
        if viz.should_quit():
            break
        arena.step()
        viz.draw()

    viz.close()
    print("Done.")


if __name__ == "__main__":
    main()
