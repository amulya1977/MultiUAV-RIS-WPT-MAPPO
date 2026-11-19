
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASE = Path("pls_mappo")

TRIAL = BASE / "results_stage7h_trial" / "training_metrics.json"
STABLE = BASE / "results_stage7h_stable" / "training_metrics.json"
EVAL = BASE / "results_stage7i_eval" / "summary.json"

OUT = BASE / "research_graphs"
OUT.mkdir(parents=True, exist_ok=True)


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


trial = load_json(TRIAL)
stable = load_json(STABLE)

if not trial or not stable:
    raise ValueError("Training metrics cannot be empty")


def values(rows, key):
    return np.array([float(row[key]) for row in rows])


def save_graph(filename):
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", OUT / filename)


def plot_reward():
    plt.figure(figsize=(10, 5))

    plt.plot(
        values(trial, "update"),
        values(trial, "mean_reward"),
        label="Actor LR=0.0003, Critic LR=0.001",
        linewidth=1.5,
        alpha=0.8
    )

    plt.plot(
        values(stable, "update"),
        values(stable, "mean_reward"),
        label="Actor LR=0.00003, Critic LR=0.0001",
        linewidth=1.5,
        alpha=0.8
    )

    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("PPO Training Update")
    plt.ylabel("Mean Reward per Agent-Step")
    plt.title("MAPPO Training Reward vs Update")
    plt.legend()
    plt.grid(alpha=0.25)
    save_graph("01_reward_vs_update.png")


def plot_smoothed_reward():
    plt.figure(figsize=(10, 5))

    for data, label in [
        (trial, "Actor 0.0003 / Critic 0.001"),
        (stable, "Actor 0.00003 / Critic 0.0001")
    ]:
        x = values(data, "update")
        y = values(data, "mean_reward")

        window = 10
        smooth = np.convolve(
            y, np.ones(window) / window, mode="valid"
        )

        plt.plot(
            x[window - 1:],
            smooth,
            label=label,
            linewidth=2
        )

    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("PPO Training Update")
    plt.ylabel("Mean Reward (10-Update Moving Average)")
    plt.title("MAPPO Reward Learning Trend")
    plt.legend()
    plt.grid(alpha=0.25)
    save_graph("02_smoothed_reward.png")


def plot_energy_efficiency():
    plt.figure(figsize=(10, 5))

    for data, label in [
        (trial, "Actor 0.0003 / Critic 0.001"),
        (stable, "Actor 0.00003 / Critic 0.0001")
    ]:
        x = values(data, "update")
        secrecy = values(data, "mean_secrecy_mbps")
        energy = values(data, "mean_energy_j")

        efficiency = np.divide(
            secrecy * 1000,
            energy,
            out=np.zeros_like(secrecy),
            where=energy > 0
        )

        plt.plot(x, efficiency, label=label, alpha=0.85)

    plt.xlabel("PPO Training Update")
    plt.ylabel("Secure Energy Efficiency (Mbits/kJ)")
    plt.title("MAPPO Secure Energy Efficiency vs Update")
    plt.legend()
    plt.grid(alpha=0.25)
    save_graph("03_energy_efficiency.png")


def plot_secrecy():
    plt.figure(figsize=(10, 5))

    for data, label in [
        (trial, "Actor 0.0003 / Critic 0.001"),
        (stable, "Actor 0.00003 / Critic 0.0001")
    ]:
        plt.plot(
            values(data, "update"),
            values(data, "mean_secrecy_mbps"),
            label=label,
            alpha=0.8
        )

    plt.xlabel("PPO Training Update")
    plt.ylabel("Mean Aggregate Secrecy Rate (Mbps)")
    plt.title("MAPPO Secrecy Rate vs Update")
    plt.legend()
    plt.grid(alpha=0.25)
    save_graph("04_secrecy_rate.png")


def plot_baseline_comparison():
    summary = load_json(EVAL)

    policies = [
        "results_stage7h_stable_mappo_update_00025",
        "results_stage7h_stable_mappo_update_00050",
        "results_stage7h_stable_mappo_update_00075",
        "results_stage7h_stable_mappo_update_00100",
        "random",
        "stationary"
    ]

    labels = [
        "MAPPO 25",
        "MAPPO 50",
        "MAPPO 75",
        "MAPPO 100",
        "Random",
        "Stationary"
    ]

    metric = "secrecy_mbits_per_kj"

    means = [summary[p][metric]["mean"] for p in policies]
    stds = [summary[p][metric]["std"] for p in policies]

    plt.figure(figsize=(11, 5))
    plt.bar(labels, means, yerr=stds, capsize=4)

    plt.ylabel("Secure Energy Efficiency (Mbits/kJ)")
    plt.title("Held-Out Secure Energy Efficiency Comparison")
    plt.xticks(rotation=20)
    plt.grid(axis="y", alpha=0.25)
    save_graph("05_baseline_energy_efficiency.png")


plot_reward()
plot_smoothed_reward()
plot_energy_efficiency()
plot_secrecy()

if EVAL.exists():
    plot_baseline_comparison()
else:
    print("Evaluation summary not found; skipped baseline comparison.")

print("\nAll available graphs generated successfully.")
