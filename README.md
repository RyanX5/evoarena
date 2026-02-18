# EvoArena

**Agent-Based Evolutionary Simulation of Emergent Game AI Behavior**

A 2D arena where autonomous agents will evolve combat and survival behaviors using genetic algorithms and neural networks. Behavior emerges through evolution — no hand-coded rules.

> **Current Status (Report 1):** Foundation milestone. Arena, agents, neural network architecture, and basic visualization are in place. Evolution and combat are planned for the next milestone.

---

## Project Structure

```
EvoArena/
├── main.py                     # Entry point (demo)
├── config.py                   # Tunable parameters
├── requirements.txt
├── simulation/
│   ├── agent.py                # Agent entity (position, movement)
│   ├── arena.py                # 2D environment and step loop
│   └── neural_network.py       # Feedforward NN architecture
├── visualization/
│   └── visualizer.py           # Real-time Pygame renderer
└── tests/
    └── test_core.py            # Unit tests
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/EvoArena.git
cd EvoArena
pip install -r requirements.txt
```

---

## Running the Demo

```bash
python main.py
```

Agents spawn in the arena and wander randomly. Press **ESC** or close the window to quit.

---

## Technologies

- **Python 3.11+**
- **NumPy** — neural network computation
- **Pygame** — real-time 2D visualization

---

## Roadmap

| Milestone | Scope |
|---|---|
| Report 1 ✅ | Arena, agents, neural network architecture, visualization |
| Report 2 | Neural-network-driven movement, combat, evolutionary loop |
| Report 3 | Fitness analysis, parameter experiments, behavioral comparisons |
| Report 4 | Advanced modes, polish, final demo |

---

*CSCI 412 – Senior Seminar I | Rohan Upadhyay*
