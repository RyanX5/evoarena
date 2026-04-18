# EvoArena - Final Report

**Rohan Upadhyay** | CSCI 411/412: Senior Seminar | April 2026

**GitHub:** https://github.com/RyanX5/evoarena  
**Live Demo:** http://seminar1.duckdns.org

---

## 1. Project Overview

### 1.1 Problem Statement and Motivation

Most approaches to building intelligent agents require writing rules by hand. You specify what the agent should do in each situation: if an enemy is close, attack; if health is low, retreat. This works fine for simple, well-defined problems, but it has an obvious ceiling - you can only encode strategies you already know about, and the resulting agents tend to be brittle outside the exact scenarios you planned for.

EvoArena takes a different approach. Instead of programming behaviors directly, the idea is to let agents figure out strategies through evolution. Each agent is controlled by a small neural network, and the weights of that network are shaped over generations by a genetic algorithm. Agents that survive and deal damage get to reproduce; agents that die quickly do not. The system never explicitly tells an agent to attack, dodge, or stay in the center - it just applies selection pressure and sees what emerges.

This project was appealing because it covers a lot of core CS ground in one place: neural networks, evolutionary algorithms, agent-based simulation, and system design. It also produces results that you can actually watch - behavioral changes across generations are visible in real time, and fitness curves give you a quantitative measure of whether the population is improving.

### 1.2 Objectives

The project had four main goals across the semester:

1. Build a working agent-based simulation with real-time rendering
2. Implement a neuroevolution loop and show that agent behavior actually changes across generations in a measurable way
3. Run systematic experiments to understand how the key parameters (mutation rate, population size, elite percentage) affect evolutionary performance
4. Deploy the simulation as a live web app that anyone can open in a browser and interact with

### 1.3 Summary of Solution

EvoArena is a Python simulation where 30 agents compete each generation in an 800x600 arena with five static obstacles. Each agent is driven by an 8-input, 12-hidden, 4-output feedforward neural network. At the end of each generation, the top 20% of agents by fitness score are selected as elites, and the next generation is produced by running uniform crossover between pairs of elites and then applying Gaussian weight mutation.

The simulation runs locally (either in a Pygame window or headless from the command line) and also as a live web application. The web version streams the simulation state over a WebSocket connection from a FastAPI backend running on a VPS, with a browser Canvas renderer on the front end and slider controls for adjusting the parameters in real time.

---

## 2. System Design and Architecture

### 2.1 Overall Architecture

The key design decision made early on was to keep the simulation completely independent of any rendering code. The arena and agents have no knowledge of Pygame or any display library - they just run physics and return state. The project has three main layers:

**Simulation core** (`simulation/`, `config.py`) - the arena, agents, and neural networks. This layer has no rendering dependency at all; it just runs physics and returns state each step.

**Frontends** - two separate options that both plug into the same core:
- Local: `visualization/visualizer.py` (Pygame window) and `visualization/fitness_graph.py` (matplotlib PNG)
- Web: `web/server.py` (FastAPI + WebSocket) streams agent state to `web/static/app.js`, which renders on a browser Canvas

**Evolution engine** (`evolution/evolve.py`) - runs between generations. Takes the surviving agent pool, applies selection, crossover, and mutation, and returns the next population.

This separation paid off when building the web version: the entire web backend required zero changes to the simulation code. The same arena and agent classes that run in the Pygame window also run on the server, just with a WebSocket broadcaster instead of a renderer.

The generation lifecycle is shown in the flowchart below:

![Simulation flowchart](https://raw.githubusercontent.com/RyanX5/evoarena/main/assets/images/screenshots/flowchart.png)

*Fig. 1: Per-generation control flow. Agents sense the environment, run a neural network forward pass, act, and accumulate fitness until the generation ends. The evolution engine then selects elites and produces the next generation.*

### 2.2 Technologies and Tools

| Layer | Technology | Purpose |
|---|---|---|
| Simulation | Python 3, NumPy | Agent logic, neural network math |
| Local visualization | Pygame | Real-time 2D rendering |
| Fitness graph | matplotlib (Agg backend) | Per-generation fitness curves |
| Web backend | FastAPI, uvicorn | HTTP server, WebSocket streaming |
| Web frontend | HTML5 Canvas, vanilla JS | Browser-based arena renderer |
| Reverse proxy | Caddy | TLS termination, domain routing |
| Deployment | Linux VPS | 24/7 server hosting |
| Testing | pytest | Unit tests for core modules |

### 2.3 Module Descriptions

| Module | Description |
|---|---|
| `config.py` | Central location for all tunable parameters |
| `simulation/neural_network.py` | 8->12->4 feedforward NN, forward pass, weight serialization |
| `simulation/agent.py` | Agent state, NN decision-making, movement, combat, fitness |
| `simulation/arena.py` | Arena environment, obstacle placement, generation step loop |
| `evolution/evolve.py` | Elitist selection, uniform crossover, Gaussian mutation |
| `visualization/visualizer.py` | Pygame real-time renderer (only display-dependent file) |
| `visualization/fitness_graph.py` | matplotlib fitness curves saved to `fitness.png` |
| `main.py` | Entry point: visual, headless, and replay modes |
| `experiments/sweep.py` | Headless parameter sensitivity sweep |
| `web/server.py` | FastAPI server, simulation thread, WebSocket broadcast |
| `web/static/app.js` | Canvas renderer, WebSocket client, live controls |

---

## 3. Implementation Details

### 3.1 Neural Network

Each agent's brain is a feedforward neural network with a single hidden layer. Both layers use tanh activation:

$$\mathbf{h} = \tanh(W_1 \mathbf{x} + \mathbf{b}_1), \quad \mathbf{o} = \tanh(W_2 \mathbf{h} + \mathbf{b}_2)$$

The architecture is $8 \to 12 \to 4$. The total number of trainable parameters is:

$$|W_1| + |b_1| + |W_2| + |b_2| = (8 \times 12) + 12 + (12 \times 4) + 4 = \mathbf{160}$$

All 160 weights are stored as a flat NumPy array. This is the unit of evolution - crossover and mutation operate directly on this flat array. Weights are initialized using Xavier initialization for $W_1$ and $W_2$, and zeros for the biases.

**Inputs (8):** All inputs are normalized to roughly $[-1, 1]$ before the forward pass.

| Index | Input | Normalization |
|---|---|---|
| 0 | Distance to nearest enemy | $\div\, d_{\max}$ (arena diagonal) |
| 1 | Angle to nearest enemy | $\div\, \pi$ |
| 2 | Own health | $\div\, 100$ |
| 3 | Nearest enemy health | $\div\, 100$ |
| 4 | Distance to nearest wall | $\div\, d_{\max}$ |
| 5 | Own velocity $v_x$ | $\div\, v_{\max}$ |
| 6 | Own velocity $v_y$ | $\div\, v_{\max}$ |
| 7 | Bias | always $1.0$ |

**Outputs (4):** The network produces four values, but only the first two are wired up to behavior:

| Index | Output | Usage |
|---|---|---|
| 0 | Move $x$ | $\times\, v_{\max} \to v_x$ |
| 1 | Move $y$ | $\times\, v_{\max} \to v_y$ |
| 2 | Attack intent | Not connected - attack is always-on |
| 3 | Flee mode | Not connected |

The decision to make attack always-on (proximity-based rather than NN-controlled) was made at the midterm to keep the fitness landscape focused on movement strategy. The idea was that if attacking was optional, the network might just learn to never attack, which is not interesting. By making it automatic, agents are forced to use positioning as their primary lever.

### 3.2 Agent Lifecycle

Each simulation step, every living agent runs through the same sequence:

1. **`decide()`** - Build the 8-input vector from the current arena state, run the NN forward pass, write $v_x$ and $v_y$ from outputs 0 and 1
2. **`move()`** - Apply velocity, bounce off arena walls by reflecting the velocity component on contact
3. **`handle_obstacles()`** - AABB vs. circle collision detection against each obstacle; push the agent out along the collision normal and reflect velocity
4. **Increment `steps_alive`**
5. **`attack()`** - Deal `ATTACK_DMG = 2` to all living enemies within `ATTACK_RANGE = 25` pixels

Dead agents are removed from `arena.agents` after each full step.

The obstacle collision uses the standard clamped-closest-point approach. For each rectangle, the closest point on the rect to the agent center is found by clamping, and if it is within `AGENT_RADIUS`, the agent is pushed out and its velocity is reflected across the surface normal:

$$\mathbf{v}' = \mathbf{v} - 2(\mathbf{v} \cdot \hat{\mathbf{n}})\hat{\mathbf{n}}$$

### 3.3 Fitness Function

Each agent accumulates a fitness score throughout its lifetime:

$$F = 2.0 \cdot D + 150.0 \cdot K + 1.0 \cdot S - 2.0 \cdot W$$

where $D$ is total damage dealt, $K$ is kill count, $S$ is steps survived, and $W$ is the accumulated wall proximity penalty.

The kill reward of $150$ was chosen to be substantially larger than the pure survival reward (at most $1.0 \times 1000 = 1000$ steps for a survivor with zero kills) to push agents toward aggressive behavior. Without a high kill weight, the easiest strategy is to just hide and outlast everyone - which does happen early in training before the wall penalty kicks in.

The wall penalty $W$ is continuous rather than binary. Each step, for each of the four walls, the agent accumulates a penalty proportional to how close it is:

$$p_t = \max\!\left(0,\; 1 - \frac{d_{\text{wall}}}{d_{\text{threshold}}}\right), \quad W = \sum_t p_t$$

The reasoning for making this continuous is explained in Section 5.1 - the first binary version was exploited almost immediately.

### 3.4 Evolution Engine

At the end of each generation, `evolution/evolve.py` builds the next population through three steps:

1. **Selection:** Sort all surviving agents by fitness (descending). Take the top 20% as elites (minimum 1 regardless of population size).

2. **Crossover:** Pick two random elites as parents. Apply uniform crossover - each weight in the child is independently drawn from parent A or parent B with equal probability:

$$c_i = \begin{cases} a_i & \text{if } m_i > 0.5 \\ b_i & \text{otherwise} \end{cases}, \quad m_i \sim \mathcal{U}(0,1)$$

3. **Mutation:** Add Gaussian noise to all weights in the child:

$$\mathbf{w}' = \mathbf{w}_{\text{child}} + \mathcal{N}(0,\, \sigma^2 \mathbf{I}), \quad \sigma = 0.1$$

Steps 2 and 3 repeat until the new population reaches `POPULATION_SIZE = 30`. All new agents start with zeroed state - position, health, and all fitness counters reset. Initial placement is random and non-overlapping, with a minimum separation of $4 \times$ agent radius enforced by up to 200 placement attempts per agent.

### 3.5 Headless Mode and Champion Save/Replay

**Headless mode** removes the Pygame dependency entirely using a lazy import pattern. When `--headless` is passed, `Visualizer` is never imported, so there is no pygame initialization and no display requirement. This was what made the parameter sweep feasible - running 9 configurations of 50 generations each with a live window open would have been too slow. The same generation loop runs in both modes; all rendering calls are guarded by `if viz:`.

**Champion save** writes the best surviving agent's weight vector to a `.npy` file after each generation:

```python
np.save(f"champions/gen_{gen:03d}.npy", agent.brain.get_weights())
```

**Replay mode** loads a saved champion and runs it against fresh random opponents each round. The main thing that needed fixing was the visual tracking: a `pinned_champion_id` attribute on `Visualizer` locks the gold ring to the loaded champion by ID, preventing it from drifting to opponents as they accumulate `steps_alive` points. This is described in more detail in Section 5.4.

![Headless mode terminal output](https://raw.githubusercontent.com/RyanX5/evoarena/main/assets/images/screenshots/headless.png)

*Fig. 2: Headless mode running 10 generations. Each generation prints best fitness, average fitness, survivor count, and confirms the champion and fitness graph saves.*

![Champion replay in arena](https://raw.githubusercontent.com/RyanX5/evoarena/main/assets/images/screenshots/champion_replay.png)

*Fig. 3: Replay mode. The loaded champion (agent 1, gold ring) competes against fresh random opponents. The gold ring stays locked to the champion via `pinned_champion_id` and does not drift to higher-fitness opponents.*

### 3.6 Web Deployment

The web version replaces Pygame with a browser Canvas, while the simulation itself runs unchanged on the server.

**Server architecture:**

```
VPS (Linux)
+-- web/server.py          FastAPI application
|   +-- GET /              Serves index.html
|   +-- WebSocket /ws      Streaming endpoint
|       +-- Simulation thread  (arena + evolution loop)
|       +-- Frame broadcaster  (60-frame bounded queue)
+-- Caddy (reverse proxy)
    +-- seminar1.duckdns.org -> localhost:8000 (HTTPS via Let's Encrypt)
```

The arena runs in a background thread at full speed, pushing agent state into a bounded queue every 3 simulation steps (roughly 20 effective fps). The WebSocket handler drains the queue and broadcasts to all connected clients. If a client falls behind, frames are dropped rather than queued indefinitely - the queue is capped at 60 frames.

**WebSocket message format:** Two message types flow from server to client:

```json
// Frame (sent every ~3 simulation steps)
{
  "type": "frame",
  "step": 42,
  "gen": 3,
  "agents": [
    {"id": 1, "x": 123.4, "y": 456.7, "health": 80.0,
     "fitness": 234.5, "is_alpha": true}
  ]
}

// Generation end (triggers fitness graph update)
{
  "type": "gen_end",
  "gen": 3,
  "best_fitness": 1245.0,
  "avg_fitness": 312.7
}
```

Obstacle data is sent once on connect since they never change. Clients can send `"config"` messages back to the server to update `population_size`, `mutation_rate`, or `elite_pct`; changes are staged and applied at the next generation boundary to avoid corrupting mid-generation state.

**Frontend:** `web/static/app.js` connects to the WebSocket on page load and automatically reconnects if the connection drops. The Canvas renderer uses the same color scheme as the Pygame version: dark background `#0f0f19`, obstacles in gray, agents in green `(80,200,120)`, and the alpha agent in yellow `(255,255,100)` with a gold ring. Health bars and agent IDs are rendered above each circle. A second canvas below the arena draws the live fitness graph using the `gen_end` messages.

**Reverse proxy:** Caddy sits in front of the FastAPI server and handles TLS termination. Caddy's automatic HTTPS via Let's Encrypt takes care of certificate management without any manual setup.

![Web app arena view](https://raw.githubusercontent.com/RyanX5/evoarena/main/assets/images/screenshots/viz_main.png)

*Fig. 4: The web app at seminar1.duckdns.org. The arena canvas streams live agent state over WebSocket. The sidebar shows the three parameter sliders (Population Size, Mutation Rate, Elite %) and a legend. The status bar at the bottom shows generation, step, agent count, and best/avg fitness.*

![Web app fitness graph](https://raw.githubusercontent.com/RyanX5/evoarena/main/assets/images/screenshots/viz_graph.png)

*Fig. 5: Live fitness graph below the arena. Best fitness (yellow) and average fitness (cyan) are plotted across all generations run so far. This is rendered on a Canvas element in the browser using the `gen_end` WebSocket messages.*

---

## 4. Results and Evaluation

### 4.1 Behavioral Evolution Across Generations

The clearest evidence that evolution is doing something useful is watching the behavior change across generations. The screenshots below document four stages of a single training run.

![Generation 1](https://raw.githubusercontent.com/RyanX5/evoarena/main/assets/images/screenshots/Gen-1.png)

*Fig. 6: Generation 1. Agents are placed randomly and move based on random initial network weights. There is no coherent strategy - agents wander around and collide with each other and obstacles unpredictably.*

![Generation 15](https://raw.githubusercontent.com/RyanX5/evoarena/main/assets/images/screenshots/Gen-15.png)

*Fig. 7: Generation 14. Wall-hugging behavior has emerged. Most agents have learned to press against arena edges, which minimizes exposure to enemies and maximizes survival time. This is a reward hacking problem - agents are optimizing the fitness function in a way that was not intended. See Section 5.1.*

![Generation 39](https://raw.githubusercontent.com/RyanX5/evoarena/main/assets/images/screenshots/Gen-39.png)

*Fig. 8: Generation 39. After the continuous wall penalty was introduced, wall-hugging stopped being profitable. The population has shifted toward arena-center positioning, with agents clustering around obstacles and engaging in actual combat.*

![Generation 97](https://raw.githubusercontent.com/RyanX5/evoarena/main/assets/images/screenshots/Gen-97.png)

*Fig. 9: Generation 97. The mature population avoids walls, moves into the center, and actively engages enemies. Agents are more spread out than in Gen 39, which probably reflects some learned spacing behavior to avoid being surrounded.*

The alpha agent (highest fitness at any given step) is highlighted in yellow with a gold ring. This makes it easier to watch the current best strategy in real time:

![Alpha agent](https://raw.githubusercontent.com/RyanX5/evoarena/main/assets/images/screenshots/alpha_marker.png)

*Fig. 10: Alpha agent (yellow, gold ring) during a mid-training generation. The alpha is near the center cluster of combat, which is consistent with the kill-focused fitness function rewarding aggressive positioning.*

### 4.2 Fitness Curves: Mutation vs. Crossover

Two 115-generation runs were recorded to measure the effect of adding crossover to the evolution engine.

**Pure mutation (single-parent reproduction):**

![Pure mutation fitness graph](https://raw.githubusercontent.com/RyanX5/evoarena/main/assets/images/screenshots/pure_mutation_graph.png)

*Fig. 11: Fitness over 115 generations under pure mutation. Best fitness jumps to 1000-1200 within the first two generations and then plateaus, oscillating in the 1000-1400 range for the rest of the run. Average fitness starts deeply negative (wall-penalty-dominated agents) and crosses zero around generation 35.*

Three phases are visible in this curve:

- **Generations 1-2:** Best fitness jumps from near zero to ~1200. One agent with favorable initial weights finds combat quickly, accumulates kills at 150 points each, and establishes the elite lineage.
- **Generations 2-35 (average negative):** Most agents are still being dominated by the wall penalty. The $-2.0 \times W$ term wipes out whatever survival and damage contributions they accumulate, producing negative total fitness. Only the elite lineage consistently scores positive.
- **Generations 35-115 (average above zero):** Wall-dominant strategies are finally eliminated as elite weights spread through the population. But the best fitness ceiling stays stuck in the 1000-1400 range - single-parent mutation can only locally perturb the existing elite weights and cannot combine distinct strategies from different lineages.

**With crossover enabled:**

![Crossover fitness graph](https://raw.githubusercontent.com/RyanX5/evoarena/main/assets/images/screenshots/crossover_graph.png)

*Fig. 12: Fitness over 115 generations with uniform crossover. The best fitness ceiling rises to around 1200-1500 and average fitness stays above zero from generation 10 onward. Both metrics show a meaningful improvement over pure mutation.*

The biggest difference is the average fitness trajectory. With crossover, population-wide improvement is faster and more stable - average fitness does not drop back to negative after crossing zero. The best fitness ceiling is also higher and more consistent. This makes sense because crossover can combine weight subsets from two different elite agents, potentially producing children that are better than either parent. The benefit does diminish over time as elites converge to similar weight distributions and crossover children start resembling single-parent offspring.

### 4.3 Parameter Sensitivity Analysis

With headless mode available, a sweep was run across three hyperparameters. Each configuration ran for 50 generations, with the two non-swept parameters held at their defaults ($N=30$, $\sigma=0.1$, elite $=20\%$).

**Mutation Rate ($\sigma \in \{0.05, 0.1, 0.2\}$):**

![Mutation rate sweep](https://raw.githubusercontent.com/RyanX5/evoarena/main/experiments/results/sweep_mutation_rate.png)

*Fig. 13: Best and average fitness across 50 generations for three mutation rates.*

| Config | Best Fitness (Gen 50) | Avg Fitness (Gen 50) |
|---|---|---|
| $\sigma = 0.05$ | ~1100 | 241.8 |
| $\sigma = 0.1$ (default) | ~1200 | 377.4 |
| $\sigma = 0.2$ | ~1350 | **395.7** |

Higher mutation provides more exploration, which pays off here because the population has not fully converged at 50 generations. $\sigma = 0.2$ edges out the default on both metrics, suggesting the default could be nudged upward. The more important result is that $\sigma = 0.05$ significantly underperforms on average fitness - with mutations that small, most of the population stays stuck near their initial weight configurations and only the elite lineage makes real progress.

**Population Size ($N \in \{15, 30, 50\}$):**

![Population size sweep](https://raw.githubusercontent.com/RyanX5/evoarena/main/experiments/results/sweep_population_size.png)

*Fig. 14: Best and average fitness across 50 generations for three population sizes.*

| Config | Best Fitness (Gen 50) | Avg Fitness (Gen 50) |
|---|---|---|
| $N = 15$ | 1096.0 | 440.6 |
| $N = 30$ (default) | **1672.5** | **751.8** |
| $N = 50$ | 1197.6 | 306.7 |

Population size has the largest effect of the three parameters, and the $N = 50$ result is counterintuitive. A larger population does not mean better evolution here. With 50 agents in the arena at once, combat becomes noisier - every agent encounters more opponents per step, so performance is more dependent on starting position and matchup luck than on actual learned strategy. The elite pool of 10 is larger in absolute terms, but it is drawn from a noisier competitive environment, which dilutes selection quality. $N = 30$ looks like a genuine optimum for this arena size, not just a default.

**Elite Percentage ($e \in \{10\%, 20\%, 30\%\}$):**

![Elite percentage sweep](https://raw.githubusercontent.com/RyanX5/evoarena/main/experiments/results/sweep_elite_pct.png)

*Fig. 15: Best and average fitness across 50 generations for three elite percentages.*

| Config | Peak Best Fitness | Avg Fitness (Gen 50) |
|---|---|---|
| $e = 10\%$ | 1154.0 | 373.0 |
| $e = 20\%$ (default) | **1619.7** | $-598.8$ |
| $e = 30\%$ | 1348.7 | $-56.0$ |

The elite percentage results show an interesting variance-stability tradeoff. 20% elitism peaks the highest but also degrades the most severely in later generations - average fitness goes negative by generation 50. The 6-agent elite pool enables diverse crossover early on, but as the elites converge to similar strategies, crossover stops being useful and the population quality degrades. 10% elitism is the most stable configuration, with positive average fitness throughout. A reasonable improvement would be to start with a larger elite percentage for diverse early exploration and narrow it as the population converges.

### 4.4 Limitations and Potential Improvements

- **Non-determinism:** `RANDOM_SEED` is defined in `config.py` but never actually applied anywhere in the code. Every run produces different results, which makes it hard to reproduce exact fitness curves.
- **Unused NN outputs:** Outputs 2 (attack intent) and 3 (flee mode) are computed by the network but ignored. Connecting them to actual behavior would let evolution explore a much larger strategy space.
- **Fixed architecture:** The 8->12->4 network was chosen by hand and never tuned. A larger hidden layer or a second hidden layer might let agents learn more complex behaviors.
- **No memory:** Agents are stateless across steps. They can only react to what they can see right now and cannot track enemy movement, remember where threats came from, or plan ahead. Recurrent architectures like LSTMs would be a natural extension.
- **Convergence ceiling:** Best fitness oscillates in a bounded range even with crossover. Techniques like fitness sharing, novelty search, or island-model evolution could help the population escape local optima.

---

## 5. Challenges and Solutions

### 5.1 Reward Hacking - Wall Hugging

The most significant challenge was reward hacking. By around generation 10, agents discovered that pressing against arena walls minimized exposure to enemies and maximized survival time. Since survival time was the dominant fitness term at that point, wall-hugging became the stable strategy.

**First fix - binary wall penalty:** A binary penalty was introduced: any agent within `WALL_DISTANCE = 150` pixels of a wall got a fixed per-step deduction. This did suppress the initial wall-hugging, but agents adapted quickly. Within a few generations, the entire population had found the exact boundary of the penalty zone and learned to sit at precisely 80px from two walls simultaneously - technically outside the penalty region, but still in a defensive corner. By generation 16, this was the dominant strategy.

**Second fix - continuous corner exposure penalty:** The penalty was redesigned to be gradual. Each step, for each of the four walls, the agent accumulates a penalty proportional to proximity:

$$p_t = \max\!\left(0,\; 1 - \frac{d_{\text{wall}}}{d_{\text{threshold}}}\right), \quad W = \sum_t p_t$$

Because there is no hard boundary, there is no threshold to exploit. Moving closer to any wall always costs more fitness. This resolved the wall-hugging problem, and Fig. 8 (Gen 39) shows the population shifting toward center positioning after the change.

### 5.2 Population Diversity Collapse

With an initial population of 10 agents and 20% elitism, only 2 elites were selected per generation. Within a few generations, all 10 agents were essentially mutations of the same 2-agent lineage. Crossover between two nearly identical parents produces children that look the same as single-parent mutation results, so diversity collapsed.

**Fix:** Population size was increased from 10 to 30, giving 6 elites per generation. The fitness improvement seen in Section 4.2 - the higher average fitness ceiling with crossover - is at least partly attributable to this change.

### 5.3 matplotlib / Pygame GIL Conflict

Adding a live fitness graph to the simulation caused a fatal crash. matplotlib's default TkAgg backend tries to open a second GUI window, which conflicts with Pygame's ownership of the main thread and triggers a GIL error at startup.

**Fix:** matplotlib's `Agg` backend is explicitly selected before any other matplotlib import. The Agg backend renders to a file buffer with no GUI component and no threading conflicts. The graph is saved to `fitness.png` after each generation.

### 5.4 Pinned Champion Visual in Replay

When replay mode was first implemented, the gold ring identifying the champion would drift off within the first 100 steps. The alpha selection logic chose the highest-fitness agent at each step; the loaded champion starts with zero accumulated fitness, so random opponents - which are accumulating `steps_alive` points immediately - overtake it quickly on the fitness ranking.

**Fix:** A `pinned_champion_id` attribute was added to `Visualizer`. When this is set, the alpha selection ignores fitness entirely and looks for the agent with that specific ID instead. The replay launcher sets this right after spawning the champion, so the ring stays locked for the entire run.

### 5.5 Web Deployment - Thread Safety and Frame Rate

Running the simulation in a background thread while simultaneously serving WebSocket clients caused two problems: (1) parameter updates from the browser could arrive mid-generation and corrupt the arena state, and (2) the simulation runs much faster than 20fps, which would cause the frame queue to grow without bound if any client was slow to drain it.

**Fix for thread safety:** Config updates from clients are staged - the server records the new values but only applies them at the generation boundary, when the evolution step creates a fresh arena with fresh state. Nothing mid-generation is modified.

**Fix for frame rate:** The queue is capped at 60 frames. The simulation thread uses `put_nowait()` and silently drops frames when the queue is full. Clients always see recent state and memory usage stays bounded regardless of how fast or slow each client is reading.

### 5.6 Lessons Learned

A few things stood out across the project:

**Reward shaping is harder than it looks.** The wall-hugging problem was the most time-consuming challenge of the semester. The first fix (binary penalty) was intuitive and reasonable, but it failed because evolution is very good at finding edge cases in a fitness function. The lesson is that any hard threshold is potentially exploitable - continuous, smooth penalties are much harder to game than binary ones.

**Architecture decisions compound.** Keeping the simulation core independent of any rendering code looked like extra work early on, but it made headless mode, champion replay, and the entire web deployment much easier to add. The web backend required zero changes to the simulation code because of this separation.

**More agents is not always better.** The $N = 50$ result in the parameter sweep was surprising. Intuitively, a larger population should mean more diversity and better evolution. But this arena has a fixed size, and adding more agents just adds noise to the fitness signal. The relationship between population size, arena complexity, and selection quality is not obvious and is worth thinking about carefully when setting up any evolutionary simulation.

**Visualization matters for debugging.** Having a real-time rendering of agent behavior made it much faster to spot problems like wall-hugging, diversity collapse, and corner exploitation. Without the visual simulation, it would have been much harder to diagnose what was going wrong from fitness numbers alone.

---

## 6. User Guide

### 6.1 Live Web Application

The simulation is publicly accessible at:

**http://seminar1.duckdns.org**

No installation required. Open the URL in any modern browser (Chrome, Firefox, Safari). The simulation starts automatically on page load.

**Controls (sidebar):**

| Control | Range | Default | Effect |
|---|---|---|---|
| Population Size | 10-80 | 30 | Number of agents per generation |
| Mutation Rate | 1-50% | 10% | Gaussian noise $\sigma$ on weight mutation |
| Elite % | 5-50% | 20% | Fraction of agents selected as parents |

Changes take effect at the start of the next generation. The fitness graph below the arena updates in real time after each generation ends.

**Legend:**
- Green circle: normal agent (health bar shown above)
- Yellow circle + gold ring: alpha agent (current highest fitness)
- Dark rectangles: static obstacles

### 6.2 Local Installation

**Requirements:** Python 3.8+

```bash
git clone https://github.com/RyanX5/evoarena.git
cd evoarena
pip install -r requirements.txt
```

**Dependencies:** `numpy`, `pygame`, `matplotlib`, `fastapi`, `uvicorn`, `websockets`

**Run modes:**

```bash
# Visual mode (Pygame window)
python main.py

# Headless mode (no window, runs fast)
python main.py --headless

# Stop after N generations
python main.py --headless --generations 50

# Replay a saved champion
python main.py --replay champions/gen_050.npy

# Run parameter sweep (saves plots to experiments/results/)
python experiments/sweep.py

# Run unit tests
python -m pytest tests/

# Start web server locally
cd web
uvicorn server:app --reload
# Open http://localhost:8000 in browser
```

### 6.3 Configuration

All simulation parameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `POPULATION_SIZE` | 30 | Agents per generation |
| `GEN_SIZE` | 1000 | Max steps per generation |
| `MUTATION_RATE` | 0.1 | Gaussian noise $\sigma$ |
| `ATTACK_RANGE` | 25 | Attack radius (pixels) |
| `ATTACK_DMG` | 2 | Damage per step per enemy in range |
| `WALL_DISTANCE` | 150 | Wall penalty threshold (pixels) |

---

## 7. Tools Usage Acknowledgment

AI tools (GitHub Copilot, Claude) were used throughout this project to assist with code suggestions, debugging, and documentation drafting. All AI-assisted outputs were reviewed, modified, and integrated with a full understanding of the implemented algorithms. This is also noted in the GitHub repository README.

This report was written in Markdown with LaTeX math in [blankt](https://blankt.app).
