"""
agent.py
Represents a single agent in the arena.

"""

import numpy as np
from simulation.neural_network import NeuralNetwork
import config

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
        self.damage_dealt = 0.0
        self.kills = 0
        self.steps_alive = 0
        self.steps_near_wall = 0

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

        # Also add to fitness parameter steps_near_wall to penalize later
        wall_dists = sorted([self.x, arena_width - self.x, self.y, arena_height - self.y])
        corner_exposure = wall_dists[0] + wall_dists[1]

        # Now proportional to wall_distance
        wall_penalty_score = 0.0
        for d in [self.x, arena_width - self.x, self.y, arena_height - self.y]:
            wall_penalty_score += max(0.0, 1.0 - (d / config.WALL_DISTANCE))
        self.steps_near_wall += wall_penalty_score

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

    def attack(self, agents: list):
        enemies = [a for a in agents if a is not self and a.alive]
        in_range = [a for a in enemies if np.sqrt((a.x - self.x)**2 + (a.y - self.y)**2) <= config.ATTACK_RANGE]

        for enemy in in_range:
            enemy.take_damage(config.ATTACK_DMG)
            self.damage_dealt += config.ATTACK_DMG

            # track kills for self
            if not enemy.alive:
                self.kills += 1


    def take_damage(self, amount: float):
        self.health -= amount
        if self.health <= 0:
            self.health = 0
            self.alive = False

    
    def fitness(self) -> float:
        """
        Returns the fitness score using weigted sum formula

        w_dmg, w_kills, w_steps, w_near_wall = 2.0, 150.0, 1.0, 1.0
        """
        w_dmg, w_kills, w_steps, w_near_wall = 2.0, 150.0, 1.0, 2.0
        return (
            self.damage_dealt * w_dmg + 
            self.kills * w_kills + 
            self.steps_alive * w_steps - 
            self.steps_near_wall * w_near_wall
            )
    
    def handle_obstacles(self, obstacles: list):
        """
        Resolves collisions between the agent (circle) and arena obstacles (rectangles).
        Uses AABB vs circle detection. finds closest point on rectangle to agent center,
        checks overlap, pushes agent out and reflects velocity along the collision normal.
        """
        for obs in obstacles:
            # Find closest point on rectangle to agent center
            closest_x = max(obs.x, min(self.x, obs.x + obs.width))
            closest_y = max(obs.y, min(self.y, obs.y + obs.height))

            # Vector from closest point to agent center
            dx = self.x - closest_x
            dy = self.y - closest_y
            distance = np.sqrt(dx**2 + dy**2)

            if distance < AGENT_RADIUS:
                # Edge case: agent center is fully inside obstacle
                if distance == 0:
                    self.y -= AGENT_RADIUS
                    self.vy *= -1
                    continue

                # Collision normal: direction to push agent out
                nx = dx / distance
                ny = dy / distance

                # Push agent out so it sits exactly on the surface
                self.x = closest_x + nx * AGENT_RADIUS
                self.y = closest_y + ny * AGENT_RADIUS

                # Reflect velocity along the normal
                dot = self.vx * nx + self.vy * ny
                self.vx -= 2 * dot * nx
                self.vy -= 2 * dot * ny
        


    def __repr__(self):
        return f"Agent(id={self.id}, pos=({self.x:.0f},{self.y:.0f}))"
