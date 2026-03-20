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
        
        # Crossover mutation
        if len(elites) >= 2:
            parent_a = random.choice(elites)
            parent_b = random.choice([e for e in elites if e is not parent_a])
            weight_a = parent_a.brain.get_weights()
            weight_b = parent_b.brain.get_weights()
            
            # Now we're crossovering 50/50 from parent a and parent b
            mask = np.random.rand(len(weight_a)) > 0.5
            child_weight = np.where(mask, weight_a, weight_b)
        else:
            elite = random.choice(elites)
            child_weight = elite.brain.get_weights()

        mutated = child_weight + np.random.normal(0, config.MUTATION_RATE, size=child_weight.shape)
        new_brain = NeuralNetwork(weights=mutated)
        mutated_agents.append(Agent(x=0, y=0, brain=new_brain))

    return mutated_agents

