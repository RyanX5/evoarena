"""
web/server.py
FastAPI backend for EvoArena.

Serves the static frontend and streams simulation state to all
connected clients via a single shared WebSocket broadcast.

Run with:
    uvicorn web.server:app --host 127.0.0.1 --port 8000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import threading
import queue as thread_queue
import asyncio
from contextlib import asynccontextmanager
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import config
from simulation.arena import Arena
from simulation.agent import Agent
from simulation.neural_network import NeuralNetwork

import numpy as np
import random


# ── Evolution helper (supports explicit elite_pct unlike evolve.py) ───────────

def _next_generation(agents, population_size, mutation_rate, elite_pct):
    agents.sort(key=lambda a: a.fitness(), reverse=True)
    elite_count = max(1, int(len(agents) * elite_pct))
    elites = agents[:elite_count]

    new_agents = []
    while len(new_agents) < population_size:
        if len(elites) >= 2:
            parent_a = random.choice(elites)
            parent_b = random.choice([e for e in elites if e is not parent_a])
            wa = parent_a.brain.get_weights()
            wb = parent_b.brain.get_weights()
            mask = np.random.rand(len(wa)) > 0.5
            child_w = np.where(mask, wa, wb)
        else:
            child_w = elites[0].brain.get_weights().copy()

        child_w = child_w + np.random.normal(0, mutation_rate, size=child_w.shape)
        new_agents.append(Agent(x=0, y=0, brain=NeuralNetwork(weights=child_w)))

    return new_agents


# ── Shared simulation state ───────────────────────────────────────────────────

class SimState:
    def __init__(self):
        self.running = False
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()

        # Config — changes applied at generation boundary
        self.population_size = config.POPULATION_SIZE
        self.mutation_rate   = config.MUTATION_RATE
        self.elite_pct       = 0.2

        # History (persists across generations for the graph)
        self.best_history: list[float] = []
        self.avg_history:  list[float] = []

        # Bounded queue: sim thread → async broadcaster
        # Frames are dropped when clients can't keep up (backpressure)
        self.frame_queue: thread_queue.Queue = thread_queue.Queue(maxsize=60)


sim = SimState()
clients: Set[WebSocket] = set()


# ── Simulation loop (runs in a background thread) ─────────────────────────────

def simulation_loop():
    Agent._id_counter = 0
    Arena._gen_count   = 0
    agents = [Agent(0, 0) for _ in range(sim.population_size)]

    sim.best_history.clear()
    sim.avg_history.clear()

    while not sim.stop_event.is_set():
        arena = Arena(width=config.ARENA_WIDTH, height=config.ARENA_HEIGHT)
        arena.reset(agents)
        Arena._gen_count += 1

        while arena.step_count < config.GEN_SIZE and len(arena.agents) > 1:
            if sim.stop_event.is_set():
                break

            arena.step()

            # Yield every 50 steps so asyncio event loop gets CPU time
            if arena.step_count % 50 == 0:
                time.sleep(0)

            # Send a frame every 3 steps (~333 frames per generation)
            if arena.step_count % 3 == 0 and arena.agents:
                best_agent = max(arena.agents, key=lambda a: a.fitness())
                frame = {
                    "type": "frame",
                    "step": arena.step_count,
                    "gen":  Arena._gen_count,
                    "agents": [
                        {
                            "id":       a.id,
                            "x":        round(a.x, 1),
                            "y":        round(a.y, 1),
                            "health":   round(a.health, 1),
                            "fitness":  round(a.fitness(), 1),
                            "is_alpha": a is best_agent,
                        }
                        for a in arena.agents
                    ],
                }
                try:
                    sim.frame_queue.put_nowait(frame)
                except thread_queue.Full:
                    pass  # drop frame — clients are catching up

        if sim.stop_event.is_set():
            break

        if not arena.agents:
            # Entire population wiped — restart fresh
            agents = [Agent(0, 0) for _ in range(sim.population_size)]
            Agent._id_counter = 0
            continue

        best = max(a.fitness() for a in arena.agents)
        avg  = sum(a.fitness() for a in arena.agents) / len(arena.agents)
        sim.best_history.append(round(best, 1))
        sim.avg_history.append(round(avg,  1))

        gen_end = {
            "type":         "gen_end",
            "gen":          Arena._gen_count,
            "best":         round(best, 1),
            "avg":          round(avg,  1),
            "survivors":    len(arena.agents),
            "best_history": sim.best_history[:],
            "avg_history":  sim.avg_history[:],
        }
        try:
            sim.frame_queue.put(gen_end, timeout=1.0)
        except thread_queue.Full:
            pass

        # Apply any config changes that arrived mid-generation
        agents = _next_generation(
            arena.agents,
            sim.population_size,
            sim.mutation_rate,
            sim.elite_pct,
        )
        Agent._id_counter = 0

    sim.running = False


# ── FastAPI app ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_broadcaster())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)

_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


async def _broadcast(message: dict):
    if not clients:
        return
    data = json.dumps(message)
    dead: Set[WebSocket] = set()
    for ws in list(clients):
        try:
            await ws.send_text(data)
        except Exception:
            dead.add(ws)
    clients.difference_update(dead)


async def _broadcaster():
    """Polls frame_queue at ~60 fps and broadcasts to all connected clients."""
    while True:
        try:
            frame = sim.frame_queue.get_nowait()
            await _broadcast(frame)
        except thread_queue.Empty:
            await asyncio.sleep(0.016)


def _obstacle_data():
    arena = Arena()
    return [
        {"x": o.x, "y": o.y, "w": o.width, "h": o.height}
        for o in arena.obstacles
    ]


def _state_msg():
    return {
        "type":    "state",
        "running": sim.running,
        "config": {
            "population_size": sim.population_size,
            "mutation_rate":   sim.mutation_rate,
            "elite_pct":       sim.elite_pct,
        },
        "obstacles":    _obstacle_data(),
        "arena":        {"width": config.ARENA_WIDTH, "height": config.ARENA_HEIGHT},
        "best_history": sim.best_history[:],
        "avg_history":  sim.avg_history[:],
    }


@app.get("/")
async def index():
    return FileResponse(os.path.join(_static_dir, "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)

    # Send current state immediately so the client can render the right UI
    await websocket.send_text(json.dumps(_state_msg()))

    try:
        while True:
            raw  = await websocket.receive_text()
            msg  = json.loads(raw)
            kind = msg.get("type")

            if kind == "start" and not sim.running:
                sim.running = True
                sim.stop_event.clear()
                sim.thread = threading.Thread(target=simulation_loop, daemon=True)
                sim.thread.start()
                await _broadcast(_state_msg())

            elif kind == "stop" and sim.running:
                sim.stop_event.set()
                # running flag cleared by the thread itself
                await _broadcast({"type": "state", "running": False,
                                   "config": _state_msg()["config"]})

            elif kind == "config":
                sim.population_size = max(5,   int(msg.get("population_size", sim.population_size)))
                sim.mutation_rate   = max(0.01, float(msg.get("mutation_rate",   sim.mutation_rate)))
                sim.elite_pct       = max(0.05, float(msg.get("elite_pct",       sim.elite_pct)))

    except WebSocketDisconnect:
        clients.discard(websocket)
