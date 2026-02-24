"""
visualizer.py
Real-time Pygame renderer for EvoArena.

Current state (Report 1):
    - Renders arena background and obstacles
    - Renders agents as colored circles with ID labels
    - Basic HUD (step counter, agent count)

Planned:
    - Health bars
    - Attack flash effects
    - Generation counter
    - Per-generation stats overlay
"""

import pygame
from simulation.agent import Agent, AGENT_RADIUS, MAX_HEALTH
from simulation.arena import Arena

BG_COLOR       = (15, 15, 25)
OBSTACLE_COLOR = (60, 60, 80)
OBSTACLE_EDGE  = (80, 80, 110)
AGENT_COLOR    = (80, 200, 120)
AGENT_OUTLINE  = (255, 255, 255)
HUD_COLOR      = (200, 200, 220)


class Visualizer:
    def __init__(self, arena: Arena, fps: int = 60):
        self.arena = arena
        self.fps = fps
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None

    def init(self):
        pygame.init()
        w = self.arena.width
        h = self.arena.height + 40   # extra space for HUD
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("EvoArena — Demo")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14)
        self.small_font = pygame.font.SysFont("monospace", 11)

    def should_quit(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        return False

    def draw(self):
        self.screen.fill(BG_COLOR)

        # Obstacles
        for obs in self.arena.obstacles:
            rect = pygame.Rect(obs.x, obs.y, obs.width, obs.height)
            pygame.draw.rect(self.screen, OBSTACLE_COLOR, rect)
            pygame.draw.rect(self.screen, OBSTACLE_EDGE, rect, 2)

        # Agents
        for agent in self.arena.agents:
            cx, cy = int(agent.x), int(agent.y)
            pygame.draw.circle(self.screen, AGENT_COLOR, (cx, cy), AGENT_RADIUS)
            pygame.draw.circle(self.screen, AGENT_OUTLINE, (cx, cy), AGENT_RADIUS, 1)
            label = self.small_font.render(str(agent.id), True, (20, 20, 20))
            self.screen.blit(label, (cx - 4, cy - 5))

        # HUD
        hud_y = self.arena.height + 8
        hud = f"Step: {self.arena.step_count}   Agents: {len(self.arena.agents)}"
        self.screen.blit(self.font.render(hud, True, HUD_COLOR), (10, hud_y))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        pygame.quit()
