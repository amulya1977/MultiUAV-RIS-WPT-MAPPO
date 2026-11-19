
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("pls_mappo/research_scalability")
OUT = Path("pls_mappo/research_graphs")
OUT.mkdir(parents=True, exist_ok=True)

# Find the scalability results
files = list(ROOT.rglob("summary.csv"))

if not files:
    raise FileNotFoundError(
        "Scalability summary.csv not found"
    )

frames = []

for path in files:
    data = pd.read_csv(path)

    if {"mode", "policy", "status"}.issubset(data.columns):
        data = data[
            (data["mode"] == "scalability") &
            (data["status"] == "ok")
        ]

        if not data.empty:
            frames.append(data)

if not frames:
    raise ValueError("No valid scalability results found")

df = pd.concat(frames, ignore_index=True)

df = df.drop_duplicates(
    subset=["case_id", "policy"],
    keep="last"
)

df["network"] = (
    df["num_uavs"].astype(str) + "U / " +
    df["num_iot"].astype(str) + "N"
)

df["network_order"] = (
    df["num_uavs"] * 1000 + df["num_iot"]
)

df = df.sort_values("network_order")

policies = ["mappo", "random", "stationary"]

networks = (
    df[["network", "network_order"]]
    .drop_duplicates()
    .sort_values("network_order")["network"]
    .tolist()
)

def compare_methods(metric, ylabel, filename):
    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(networks))
    width = 0.25

    for i, policy in enumerate(policies):
        subset = df[df["policy"] == policy].copy()
        subset = subset.set_index("network")

        missing = [n for n in networks if n not in subset.index]
        if missing:
            raise ValueError(
                f"Missing {policy} data for {missing}"
            )

        means = [
            float(subset.loc[n, metric + "_mean"])
            for n in networks
        ]

        stds = [
            float(subset.loc[n, metric + "_std"])
            for n in networks
        ]

        positions = x + (i - 1) * width

        ax.bar(
            positions,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=policy.upper()
        )

    ax.set_xticks(x)
    ax.set_xticklabels(networks)

    ax.set_xlabel(
        "Network Configuration (UAVs / IoT Nodes)"
    )
    ax.set_ylabel(ylabel)
    ax.set_title(
        ylabel + " — Method Comparison"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(
        OUT / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("Saved:", OUT / filename)


# Graph 17: Secrecy comparison
compare_methods(
    "secrecy_mbps",
    "Mean Secrecy Rate (Mbps)",
    "17_reliability_secrecy.png"
)

# Graph 18: UAV energy comparison
compare_methods(
    "uav_energy_j_per_step",
    "UAV Energy Consumption (J/Step)",
    "18_reliability_energy.png"
)

# Graph 19: Energy efficiency
compare_methods(
    "secrecy_mbits_per_kj",
    "Secure Energy Efficiency (Mbits/kJ)",
    "19_reliability_efficiency.png"
)

# Graph 20: IoT battery depletion
compare_methods(
    "final_depleted_iot_fraction",
    "Final Depleted IoT Fraction",
    "20_reliability_battery.png"
)

# Graph 21: Collision comparison
compare_methods(
    "collision_reversions_per_step",
    "Collision Reversions per Step",
    "21_reliability_collisions.png"
)

print("\nAll five reliability graphs generated!")
print("Output folder:", OUT)
