"""
arena.py
The 2D simulation environment.

"""

import numpy as np
from simulation.agent import Agent, AGENT_RADIUS


class Obstacle:
    """A static rectangular obstacle."""
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


class Arena:
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.agents: list[Agent] = []
        self.step_count = 0
        self.rng = np.random.default_rng()

        self.obstacles = [
            Obstacle(200, 150, 60, 60),
            Obstacle(540, 150, 60, 60),
            Obstacle(200, 390, 60, 60),
            Obstacle(540, 390, 60, 60),
            Obstacle(370, 270, 60, 60),
        ]

    def reset(self, agents: list[Agent]):
        """Place agents at random positions for a new round."""
        self.agents = agents
        self.step_count = 0
        self._place_agents_randomly()

    def _place_agents_randomly(self):
        """Scatter agents at random non-overlapping positions."""
        placed = []
        for agent in self.agents:
            for _ in range(200):
                x = self.rng.integers(AGENT_RADIUS + 20, self.width - AGENT_RADIUS - 20)
                y = self.rng.integers(AGENT_RADIUS + 20, self.height - AGENT_RADIUS - 20)
                too_close = any(
                    np.sqrt((x - px)**2 + (y - py)**2) < AGENT_RADIUS * 4
                    for px, py in placed
                )
                if not too_close:
                    agent.x, agent.y = float(x), float(y)
                    placed.append((x, y))
                    break

    def step(self):
        """Advance simulation by one time step."""
        for agent in self.agents:
            agent.decide(self.agents, self.width, self.height)
            agent.move(self.width, self.height)
        self.step_count += 1
