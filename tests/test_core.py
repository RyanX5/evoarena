"""
test_core.py
Unit tests for Report 1 components.
Run with: python -m pytest tests/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from simulation.neural_network import NeuralNetwork
from simulation.agent import Agent, MAX_HEALTH, AGENT_RADIUS
from simulation.arena import Arena


class TestNeuralNetwork:
    def test_output_shape(self):
        nn = NeuralNetwork()
        out = nn.forward(np.zeros(NeuralNetwork.INPUT_SIZE))
        assert out.shape == (NeuralNetwork.OUTPUT_SIZE,)

    def test_output_range(self):
        nn = NeuralNetwork()
        out = nn.forward(np.random.randn(NeuralNetwork.INPUT_SIZE))
        assert np.all(out >= -1) and np.all(out <= 1)

    def test_weights_roundtrip(self):
        nn = NeuralNetwork()
        w = nn.get_weights()
        nn2 = NeuralNetwork(weights=w)
        np.testing.assert_array_equal(nn.get_weights(), nn2.get_weights())

    def test_copy_is_independent(self):
        nn = NeuralNetwork()
        copy = nn.copy()
        copy.W1[0, 0] += 999
        assert nn.W1[0, 0] != copy.W1[0, 0]


class TestAgent:
    def test_initial_state(self):
        a = Agent(100, 100)
        assert a.health == MAX_HEALTH
        assert a.alive is True

    def test_moves_within_bounds(self):
        arena = Arena(800, 600)
        a = Agent(400, 300)
        rng = np.random.default_rng(0)
        for _ in range(200):
            a.wander(rng)
            a.move(arena.width, arena.height)
            assert AGENT_RADIUS <= a.x <= arena.width - AGENT_RADIUS
            assert AGENT_RADIUS <= a.y <= arena.height - AGENT_RADIUS

    def test_has_brain(self):
        a = Agent(0, 0)
        assert isinstance(a.brain, NeuralNetwork)


class TestArena:
    def test_agents_placed_in_bounds(self):
        arena = Arena(800, 600)
        agents = [Agent(0, 0) for _ in range(8)]
        arena.reset(agents)
        for a in agents:
            assert AGENT_RADIUS <= a.x <= arena.width - AGENT_RADIUS
            assert AGENT_RADIUS <= a.y <= arena.height - AGENT_RADIUS

    def test_step_increments_counter(self):
        arena = Arena(800, 600)
        arena.reset([Agent(0, 0) for _ in range(4)])
        arena.step()
        assert arena.step_count == 1

    def test_obstacles_present(self):
        arena = Arena()
        assert len(arena.obstacles) > 0
