
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("pls_mappo")
OUT = ROOT / "research_graphs"
OUT.mkdir(parents=True, exist_ok=True)

files = []
for path in ROOT.rglob("summary.csv"):
    try:
        cols = pd.read_csv(path, nrows=1).columns
        if {"actor_lr", "critic_lr", "minibatch_size",
            "secrecy_mbps_mean", "policy"}.issubset(cols):
            files.append(path)
    except Exception:
        pass

if not files:
    raise FileNotFoundError(
        "Cannot find Stage 8 sensitivity summary.csv"
    )

source = max(files, key=lambda p: p.stat().st_mtime)
print("Reading:", source)

df = pd.read_csv(source)

df = df[
    (df["mode"] == "sensitivity") &
    (df["policy"] == "mappo") &
    (df["status"] == "ok")
].copy()

if df.empty:
    raise ValueError("No successful MAPPO sensitivity results")

def lr_label(row):
    if abs(row["actor_lr"] - 0.00003) < 1e-9:
        return "Stable reference"
    return f'LR {row["actor_lr"]:g}'

df["lr_label"] = df.apply(lr_label, axis=1)

order = ["Stable reference", "LR 0.01",
         "LR 0.03", "LR 0.05"]

metrics = [
    (
        "secrecy_mbps_mean",
        "Mean Secrecy Rate (Mbps)",
        "06_batch_vs_secrecy.png"
    ),
    (
        "reward_per_step_mean",
        "Mean Reward per Step",
        "07_batch_vs_reward.png"
    ),
    (
        "secrecy_mbits_per_kj_mean",
        "Secure Energy Efficiency (Mbits/kJ)",
        "08_batch_vs_efficiency.png"
    ),
    (
        "final_depleted_iot_fraction_mean",
        "Final Depleted IoT Fraction",
        "09_batch_vs_depletion.png"
    )
]

for metric, ylabel, filename in metrics:
    plt.figure(figsize=(9, 5))

    for label in order:
        subset = df[df["lr_label"] == label]
        subset = subset.sort_values("minibatch_size")

        if subset.empty:
            continue

        plt.plot(
            subset["minibatch_size"],
            subset[metric],
            marker="o",
            linewidth=2,
            label=label
        )

    if "reward" in metric:
        plt.axhline(0, color="black",
                    linestyle="--", linewidth=1)

    plt.xlabel("PPO Minibatch Size")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs Minibatch Size")
    plt.xticks([128, 256, 512])
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=300)
    plt.close()

    print("Saved:", OUT / filename)

print("All sensitivity comparison graphs generated.")
