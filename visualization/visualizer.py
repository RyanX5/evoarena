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

        # finding the "alpha"
        best_agent = None
        max_fitness = -float('inf')
        
        if self.arena.agents:
            for agent in self.arena.agents:
                f = agent.fitness()
                if f > max_fitness:
                    max_fitness = f
                    best_agent = agent

        # Obstacles
        for obs in self.arena.obstacles:
            rect = pygame.Rect(obs.x, obs.y, obs.width, obs.height)
            pygame.draw.rect(self.screen, OBSTACLE_COLOR, rect)
            pygame.draw.rect(self.screen, OBSTACLE_EDGE, rect, 2)

        # Agents
        for agent in self.arena.agents:

            # For healthbar
            bar_width = 20
            bar_height = 4
            bar_x = int(agent.x) - bar_width // 2
            bar_y = int(agent.y) - AGENT_RADIUS - 8
            bar_color = (180, 0, 0) # Red
            bar_fill_color = (0, 200, 0) # Green

            cx, cy = int(agent.x), int(agent.y)
            
            # Draw the alpha ring and determine agent color
            if agent == best_agent:
                # Draw gold ring first so it is behind the health bar but visible around the agent
                pygame.draw.circle(self.screen, (255, 215, 0), (cx, cy), AGENT_RADIUS + 4, 3)
                current_agent_color = (255, 255, 100)
            else:
                current_agent_color = AGENT_COLOR

            # Draw agent body
            pygame.draw.circle(self.screen, current_agent_color, (cx, cy), AGENT_RADIUS)
            pygame.draw.circle(self.screen, AGENT_OUTLINE, (cx, cy), AGENT_RADIUS, 1)
            
            # Draw health bar background
            pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, bar_width, bar_height))
            
            # Draw ID label
            label = self.small_font.render(str(agent.id), True, (20, 20, 20))
            self.screen.blit(label, (cx - 4, cy - 5))

            # health bar fill with green
            fill_width = int(bar_width * (agent.health/MAX_HEALTH))
            pygame.draw.rect(self.screen, bar_fill_color, (bar_x, bar_y, fill_width, bar_height))

        # HUD
        hud_y = self.arena.height + 8
        best_fit_val = f"{max_fitness:.1f}" if best_agent else "0.0"
        hud = f"Step: {self.arena.step_count}   Agents: {len(self.arena.agents)}    Gen: {Arena._gen_count}    Best Fit: {best_fit_val}"
        self.screen.blit(self.font.render(hud, True, HUD_COLOR), (10, hud_y))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        max_fitness_score = -float('inf')
        max_score_agent = None
        
        if self.arena.agents:
            for agent in self.arena.agents:
                curr_fitness = agent.fitness()
                if curr_fitness > max_fitness_score:
                    max_fitness_score = curr_fitness
                    max_score_agent = agent
            
            if max_score_agent:
                print(f"Max fitness score: {max_fitness_score:.2f} by Agent id: {max_score_agent.id}")
        else:
            print("No agents remaining in the arena.")
            
        pygame.quit()