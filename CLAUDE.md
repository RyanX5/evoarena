# EvoArena — CLAUDE.md

Comprehensive technical reference for the EvoArena codebase. Keep this up to date as the project evolves.

---

## Project Summary

EvoArena is an agent-based evolutionary simulation in Python. Autonomous agents compete in a 2D arena and develop combat/survival behaviors through a genetic algorithm. Each agent is driven by a small feedforward neural network whose weights are evolved across generations.

**Course:** CSCI 411/412 Senior Seminar (Dr. Qi Li)  
**Status:** All core systems complete as of Report 3 (April 2026). Next: web deployment + final report.

---

## File Structure

```
evoarena/
├── config.py                  # Central config (all tunable params)
├── main.py                    # Entry point — visual, headless, replay modes
├── simulation/
│   ├── arena.py               # Arena environment + step loop
│   ├── agent.py               # Agent class — movement, combat, fitness, NN input/output
│   └── neural_network.py      # 8→12→4 feedforward NN, weight serialization
├── evolution/
│   └── evolve.py              # Elitist selection, crossover, Gaussian mutation
├── visualization/
│   ├── visualizer.py          # Pygame renderer (only display-dependent file)
│   └── fitness_graph.py       # matplotlib fitness graph, saved to fitness.png
├── experiments/
│   └── sweep.py               # Parameter sensitivity sweep (headless)
├── tests/
│   └── test_core.py           # Unit tests (NN, Agent, Arena)
├── docs/                      # Assignment reports (PDFs)
└── README.md                  # Outdated — needs rewrite for final submission
```

---

## config.py — All Parameters

| Constant | Value | Description |
|---|---|---|
| `ARENA_WIDTH` | 800 | Arena width in pixels |
| `ARENA_HEIGHT` | 600 | Arena height in pixels |
| `POPULATION_SIZE` | 30 | Agents per generation |
| `GEN_SIZE` | 1000 | Max steps per generation |
| `FPS` | 60 | Pygame frame rate |
| `SCALE` | 1.0 | Display scale (unused) |
| `RANDOM_SEED` | 42 | Not actually applied anywhere currently |
| `ATTACK_RANGE` | 25 | Radius within which attacks hit |
| `ATTACK_DMG` | 2 | Damage dealt per step per enemy in range |
| `MUTATION_RATE` | 0.1 | Gaussian noise σ for weight mutation |
| `WALL_DISTANCE` | 150 | Proximity threshold for wall penalty |

---

## Neural Network (simulation/neural_network.py)

**Architecture:** 8 → 12 → 4, both layers use tanh activation  
**Weight count:** 8×12 + 12 + 12×4 + 4 = **160 parameters** (flat numpy array)  
**Initialization:** Xavier for W1, W2; zeros for b1, b2  
**Serialization:** `get_weights()` → flat 1D array; `set_weights(arr)` restores; saved as `.npy` files

**8 Inputs (all normalized):**
| Index | Input | Normalization |
|---|---|---|
| 0 | Distance to nearest enemy | / max_dist (diagonal of arena) |
| 1 | Angle to nearest enemy | / π (range: -1 to 1) |
| 2 | Own health | / MAX_HEALTH (100) |
| 3 | Nearest enemy health | / MAX_HEALTH |
| 4 | Distance to nearest wall | / max_dist |
| 5 | Own velocity x | / MAX_SPEED (2.0) |
| 6 | Own velocity y | / MAX_SPEED |
| 7 | Bias | always 1.0 |

**4 Outputs:**
| Index | Output | Usage |
|---|---|---|
| 0 | Move x | × MAX_SPEED → vx |
| 1 | Move y | × MAX_SPEED → vy |
| 2 | Attack intent | **NOT IMPLEMENTED** — attack is always-on |
| 3 | Flee mode | **NOT IMPLEMENTED** |

---

## Agent (simulation/agent.py)

**Constants:** `AGENT_RADIUS = 10`, `MAX_HEALTH = 100.0`, `MAX_SPEED = 2.0`

**State per agent:** `id`, `x`, `y`, `vx`, `vy`, `health`, `alive`, `damage_dealt`, `kills`, `steps_alive`, `steps_near_wall`, `brain`

**Per-step lifecycle (called by Arena.step):**
1. `decide()` — build 8-input vector, NN forward pass, set vx/vy
2. `move()` — apply velocity, bounce off walls (reflect velocity)
3. `handle_obstacles()` — AABB vs circle collision, push out + reflect velocity
4. `steps_alive += 1`
5. `attack()` — deal ATTACK_DMG to all alive enemies within ATTACK_RANGE each step (always on, not NN-controlled)
6. Dead agents removed from arena.agents list

**Fitness formula:**
```
fitness = damage_dealt × 2.0
        + kills × 150.0
        + steps_alive × 1.0
        - steps_near_wall × 2.0
```
`steps_near_wall` is a continuous penalty: each step, for each wall within 150px, adds `max(0, 1 - dist/150)` — proportional, not binary.

**Wall penalty detail:** Uses corner exposure (`wall_dists[0] + wall_dists[1]`) to compute penalty score, but the actual penalty added is the sum across all 4 walls individually.

---

## Arena (simulation/arena.py)

**Obstacles:** 5 hardcoded 60×60 rectangles:
- (200, 150), (540, 150) — top row
- (200, 390), (540, 390) — bottom row  
- (370, 270) — center

**Generation loop:**
- Runs until `step_count >= 1000` OR `len(arena.agents) <= 1`
- `Arena._gen_count` is a class-level counter (persists across Arena instances)

**Agent placement:** Random, non-overlapping (min 4× radius separation), up to 200 attempts per agent

---

## Evolution (evolution/evolve.py)

**`next_generation(agents)`:**
1. Sort all surviving agents by fitness (descending)
2. Take top 20% as elites (min 1) — hardcoded 20%, not configurable via this function
3. Build new population of `config.POPULATION_SIZE` agents:
   - Pick 2 random elites as parents
   - Crossover: 50/50 random weight mask (each weight independently from parent A or B)
   - Gaussian mutation: add `N(0, config.MUTATION_RATE)` to all weights
4. All new agents start with fresh state (x=0, y=0, fitness stats zeroed)

**Note:** `sweep.py` has its own `next_generation_with_elite()` that accepts explicit params — used only in the sweep script.

---

## main.py — CLI Modes

```bash
python main.py                              # Visual mode (Pygame window)
python main.py --headless                   # No window, runs indefinitely
python main.py --headless --generations 50  # Stop after 50 gens
python main.py --replay champions/gen_050.npy  # Watch saved champion
```

**Champions:** Saved to `champions/gen_NNN.npy` after every generation (best surviving agent's weights)

**Replay mode:** Loads champion weights, pits against fresh random opponents each round. `viz.pinned_champion_id` locks the gold ring to the champion agent regardless of fitness.

**Headless mode:** `Visualizer` import is skipped entirely. Console prints `best=X avg=Y survivors=Z` per generation.

---

## Visualizer (visualization/visualizer.py)

**Colors:** BG `#0f0f19`, obstacles `(60,60,80)`, agents green `(80,200,120)`, alpha agent yellow `(255,255,100)` with gold ring  
**HUD:** Bottom bar shows `Step | Agents | Gen | Best Fit`  
**Alpha agent:** Highest-fitness agent gets gold ring. In replay mode, `pinned_champion_id` overrides fitness-based selection.  
**Only Pygame-dependent file** — isolation point for web conversion.

---

## experiments/sweep.py

Sweeps one parameter at a time, others held at defaults:
- Mutation rate: [0.05, 0.1, 0.2]
- Population size: [15, 30, 50]
- Elite pct: [0.1, 0.2, 0.3]

50 generations per config. Results saved to `experiments/results/sweep_*.json` and `sweep_*.png`.

**Key findings:**
- N=30 is the genuine optimum for this arena/gen-length (N=50 hurts — too chaotic)
- σ=0.2 slightly edges out σ=0.1 (population hasn't fully converged at 50 gens)
- 20% elitism peaks highest but degrades late; 10% is most stable

---

## Known Issues / TODOs

- `RANDOM_SEED` in config is defined but never seeded anywhere
- NN outputs[2] (attack) and outputs[3] (flee) are defined but never used — attack is always-on proximity-based
- README.md is outdated (says agents move randomly — written in Report 1)
- `wander()` method in agent.py is vestigial (was used before NN was wired up)
- `corner_exposure` variable computed in `decide()` but not used anywhere

---

## Web Deployment Plan (Option A — In Progress)

Goal: Replace Pygame with a browser Canvas renderer. Simulation logic stays pure Python on VPS.

**Stack:**
- Backend: FastAPI + WebSockets (Python, on VPS)
- Frontend: HTML/CSS/JS with Canvas API
- The simulation runs server-side; each frame sends agent state JSON to the browser

**WebSocket payload per frame:**
```json
{
  "step": 42,
  "gen": 3,
  "agents": [
    {"id": 1, "x": 123.4, "y": 456.7, "health": 80.0, "fitness": 234.5, "is_alpha": true},
    ...
  ]
}
```

**Obstacles are static** — send once on connect, not every frame.

**Key isolation point:** `visualization/visualizer.py` is the only Pygame file. Everything else (arena, agent, NN, evolve) runs headlessly with no changes needed.
