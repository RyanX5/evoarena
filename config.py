"""
config.py
Central configuration for EvoArena.
Edit values here to change simulation behavior without touching core code.
"""

# ── Arena ─────────────────────────────────────────────────────────────────────
ARENA_WIDTH  = 800
ARENA_HEIGHT = 600

# ── Population ────────────────────────────────────────────────────────────────
POPULATION_SIZE = 30
GEN_SIZE = 1000

# ── Display ──────────────────────────────────────────────────────────────────
FPS   = 9999
SCALE = 1.0

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

# Attack
ATTACK_RANGE = 25
ATTACK_DMG = 2

# Mutation
MUTATION_RATE = 0.1

# Penalties
WALL_DISTANCE = 50
