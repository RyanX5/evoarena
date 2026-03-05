"""
agent.py
Represents a single agent in the arena.

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

    def decide(self, agents: list, arena_width: int, arena_height: int):
        """
        Builds an 8-input vector and passes to the NN with forward()
        The 8-input vector is defined in neural_network.py
        """

        input_vector = []
        max_dist = np.sqrt(arena_width**2 + arena_height**2)

        # First, finding the nearest enemy and the distance
        enemies = [a for a in agents if a is not self and a.alive]
        nearest = None
        nearest_dist = float('inf')

        for enemy in enemies:
            d = np.sqrt((enemy.x - self.x)**2 + (enemy.y - self.y)**2)
            if d < nearest_dist:
                nearest_dist = d
                nearest = enemy

        if nearest is None:
            inputs = np.zeros(8)
            inputs[7] = 1.0
            outputs = self.brain.forward(inputs)
            self.vx = float(outputs[0]) * MAX_SPEED
            self.vy = float(outputs[1]) * MAX_SPEED
            return
        
        input_vector.append(nearest_dist/max_dist)

        # Now calculate the angle with the nearest enemy
        nearest_angle = np.arctan2(nearest.y - self.y, nearest.x - self.x)
        input_vector.append(nearest_angle/np.pi)

        # Get own health
        own_health = self.health / MAX_HEALTH
        input_vector.append(own_health)

        # Get enemy health
        nearest_health = nearest.health / MAX_HEALTH
        input_vector.append(nearest_health)

        # Distance to nearest wall
        nearest_wall_dist = min(self.x, arena_width - self.x, self.y, arena_height - self.y)
        input_vector.append(nearest_wall_dist/max_dist)

        # Velocity x
        vel_x = self.vx
        input_vector.append(vel_x/MAX_SPEED)

        # Velocity y 
        vel_y = self.vy
        input_vector.append(vel_y/MAX_SPEED)

        # bias
        bias = 1.0
        input_vector.append(bias)

        # Now doing the forward pass, and getting output
        inputs = np.array(input_vector)
        outputs = self.brain.forward(inputs=inputs)

        # Now setting the dy and dv according to the NN output
        self.vx = float(outputs[0] * MAX_SPEED)
        self.vy = float(outputs[1] * MAX_SPEED)

        # TODO: outputs[2] = attack intent (unused until combat is implemented)
        # TODO: outputs[3] = flee mode (unused until combat is implemented)


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
