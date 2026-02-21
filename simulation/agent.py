"""
agent.py
Represents a single agent in the arena.

Current state (Report 1):
    - Position, velocity, health
    - Random wandering movement (neural network control planned for Report 2)
    - No combat yet (planned for Report 2)

Planned:
    - Neural-network-driven decisions
    - Combat mechanics (attack, damage, death)
    - Fitness scoring
"""

import numpy as np
from simulation.neural_network import NeuralNetwork

AGENT_RADIUS = 10
MAX_HEALTH   = 100.0
MAX_SPEED    = 2.0


class Agent:
    _id_counter = 0

    def __init__(self, x: float, y: float, brain: NeuralNetwork = None):
        Agent._id_counter += 1
        self.id = Agent._id_counter

        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0

        self.health = MAX_HEALTH
        self.alive = True

        # Brain is initialized but not yet used for decisions
        self.brain = brain if brain is not None else NeuralNetwork()

    def wander(self, rng: np.random.Generator):
        """
        Temporary random movement for demo purposes.
        Will be replaced by neural network decisions in Report 2.
        """
        # Small random nudge each step
        self.vx += rng.uniform(-0.5, 0.5)
        self.vy += rng.uniform(-0.5, 0.5)

        # Clamp speed
        speed = np.sqrt(self.vx**2 + self.vy**2)
        if speed > MAX_SPEED:
            self.vx = (self.vx / speed) * MAX_SPEED
            self.vy = (self.vy / speed) * MAX_SPEED

    def move(self, arena_width: int, arena_height: int):
        """Apply velocity and bounce off arena walls."""
        self.x += self.vx
        self.y += self.vy

        # Bounce off walls
        if self.x < AGENT_RADIUS:
            self.x = AGENT_RADIUS
            self.vx *= -1
        if self.x > arena_width - AGENT_RADIUS:
            self.x = arena_width - AGENT_RADIUS
            self.vx *= -1
        if self.y < AGENT_RADIUS:
            self.y = AGENT_RADIUS
            self.vy *= -1
        if self.y > arena_height - AGENT_RADIUS:
            self.y = arena_height - AGENT_RADIUS
            self.vy *= -1

    def __repr__(self):
        return f"Agent(id={self.id}, pos=({self.x:.0f},{self.y:.0f}))"
