"""
fitness_graph.py
Fitness graph using matplotlib.
Saves best and average fitness across generations to fitness.png after each generation.
Open fitness.png in an image viewer alongside the arena window for a live-ish view.
"""

import matplotlib
matplotlib.use("Agg")  # no GUI — renders to file, no conflict with pygame
import matplotlib.pyplot as plt


class FitnessGraph:

    def __init__(self, path: str = "fitness.png"):
        self.path = path

    def update(self, best_history: list, avg_history: list):
        """Redraw and save the graph with the latest generation data."""
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#0f0f19")
        ax.set_facecolor("#0f0f19")

        generations = list(range(1, len(best_history) + 1))

        ax.plot(generations, best_history, color="#ffdc32", linewidth=2, label="best")
        ax.plot(generations, avg_history,  color="#50c8c8", linewidth=2, label="avg")

        # Dot on the latest point
        ax.plot(generations[-1], best_history[-1], "o", color="#ffdc32", markersize=5)
        ax.plot(generations[-1], avg_history[-1],  "o", color="#50c8c8", markersize=5)

        ax.set_title(f"Fitness — Generation {len(best_history)}", color="#c8c8dc", fontsize=11)
        ax.set_xlabel("Generation", color="#c8c8dc")
        ax.set_ylabel("Fitness",    color="#c8c8dc")
        ax.tick_params(colors="#a0a0b4")
        ax.spines["bottom"].set_color("#a0a0b4")
        ax.spines["left"].set_color("#a0a0b4")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#404060")

        fig.tight_layout()
        fig.savefig(self.path, facecolor="#0f0f19", dpi=120)
        plt.close(fig)
        print(f"  [graph] saved to {self.path}  (best={best_history[-1]:.1f}, avg={avg_history[-1]:.1f})")