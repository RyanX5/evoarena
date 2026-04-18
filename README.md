# EvoArena

An agent-based evolutionary simulation where autonomous agents compete in a 2D arena and develop combat and survival strategies through a genetic algorithm. Each agent is controlled by a feedforward neural network whose weights are evolved over generations - no behaviors are hand-coded.

**Live demo:** http://seminar1.duckdns.org

![EvoArena](https://raw.githubusercontent.com/RyanX5/evoarena/main/assets/images/screenshots/alpha_marker.png)

---

## How It Works

- 30 agents compete each generation in an 800x600 arena with five obstacles
- Each agent's brain is an 8-input, 12-hidden, 4-output neural network
- Agents sense nearby enemies, their own health, wall distances, and velocity
- At the end of each generation, the top 20% by fitness reproduce via crossover + Gaussian mutation
- Fitness rewards kills (150 pts), damage dealt (2x), and survival time (1 pt/step), and penalizes wall proximity

Behaviors like center-arena clustering, active combat, and enemy tracking emerge from selection pressure alone over ~30-50 generations.

---

## Setup

**Requirements:** Python 3.8+

```bash
git clone https://github.com/RyanX5/evoarena.git
cd evoarena
pip install -r requirements.txt
```

---

## Running

```bash
# Visual mode - Pygame window with live rendering
python main.py

# Headless mode - no window, runs fast
python main.py --headless

# Headless for N generations then stop
python main.py --headless --generations 50

# Replay a saved champion against random opponents
python main.py --replay champions/gen_050.npy

# Parameter sensitivity sweep (saves plots to experiments/results/)
python experiments/sweep.py

# Unit tests
python -m pytest tests/

# Web server (local)
cd web && uvicorn server:app --reload
# then open http://localhost:8000
```

---

## Web App

The simulation runs as a live web application at **http://seminar1.duckdns.org**.

- Arena streams over WebSocket from a FastAPI backend
- Browser Canvas renderer matches the Pygame version
- Sidebar sliders for population size, mutation rate, and elite % (changes apply next generation)
- Live fitness graph updates after each generation

Stack: FastAPI + uvicorn on a Linux VPS, Caddy as reverse proxy with automatic HTTPS.

---

## Project Structure

```
evoarena/
├── config.py                  # All tunable parameters
├── main.py                    # Entry point (visual / headless / replay)
├── simulation/
│   ├── arena.py               # Arena environment and step loop
│   ├── agent.py               # Agent logic, combat, fitness
│   └── neural_network.py      # 8->12->4 feedforward NN
├── evolution/
│   └── evolve.py              # Selection, crossover, mutation
├── visualization/
│   ├── visualizer.py          # Pygame renderer
│   └── fitness_graph.py       # matplotlib fitness curves
├── experiments/
│   └── sweep.py               # Parameter sensitivity sweep
├── web/
│   ├── server.py              # FastAPI + WebSocket backend
│   └── static/                # Frontend (HTML, JS, CSS)
├── tests/
│   └── test_core.py           # Unit tests
└── docs/                      # Progress reports
```

---

## Key Parameters (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `POPULATION_SIZE` | 30 | Agents per generation |
| `GEN_SIZE` | 1000 | Max steps per generation |
| `MUTATION_RATE` | 0.1 | Gaussian noise sigma |
| `ATTACK_RANGE` | 25 | Attack radius in pixels |
| `ATTACK_DMG` | 2 | Damage per step per enemy in range |
| `WALL_DISTANCE` | 150 | Wall penalty threshold in pixels |

---

## AI Tools

AI tools (GitHub Copilot, Claude) were used throughout this project to assist with code suggestions, debugging, and documentation. All outputs were reviewed, understood, and integrated manually.

---

**Course:** CSCI 411/412 Senior Seminar - Dr. Qi Li
