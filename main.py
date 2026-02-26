"""
main.py
EvoArena entry point — Report 1 demo.

Launches the arena with randomly wandering agents.
No evolution or combat yet — this demonstrates the foundation:
    - Arena environment with obstacles
    - Agent spawning and movement
    - Real-time Pygame visualization

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
