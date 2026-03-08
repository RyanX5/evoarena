"""
This is the evolution engine.

It takes the 20% best agents from each generation and starts a new generation with those values as weights
Also handles mutation.
"""

import numpy as np
import config
from simulation.neural_network import NeuralNetwork
import random
from simulation.agent import Agent


def next_generation(agents: list) -> list:
    """
    Takes the top 20% best agents from the given list and returns
    the elite list with some mutation. Mutation is done by
    slightly tuning the weight with some Gaussian noise.
    """
    agents.sort(key=lambda a: a.fitness(), reverse=True)

    # Now get the top 20% elites
    elite_count = max(1, int(len(agents) * 0.2))
    elites = agents[:elite_count]

    # Do some mutation and create a new mutated pool
    mutated_agents = []
    
    while len(mutated_agents) < config.POPULATION_SIZE:
        elite = random.choice(elites)
        weights = elite.brain.get_weights()
        mutated = weights + np.random.normal(0, config.MUTATION_RATE, size=weights.shape)
        new_brain = NeuralNetwork(weights=mutated)
        mutated_agents.append(Agent(x=0, y=0, brain=new_brain))

    return mutated_agents

