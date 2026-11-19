
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("pls_mappo/research_scalability")
OUT = Path("pls_mappo/research_graphs")
OUT.mkdir(parents=True, exist_ok=True)

files = list(ROOT.rglob("summary.csv"))

if not files:
    raise FileNotFoundError("No scalability summary.csv found")

df = pd.concat(
    [pd.read_csv(f) for f in files],
    ignore_index=True
)

df = df[
    (df["mode"] == "scalability") &
    (df["status"] == "ok")
].copy()

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

def plot_metric(metric, ylabel, filename, policies):
    plt.figure(figsize=(9, 5))

    for policy in policies:
        subset = df[df["policy"] == policy]
        subset = subset.sort_values("network_order")

        plt.plot(
            subset["network"],
            subset[metric],
            marker="o",
            linewidth=2,
            label=policy.upper()
        )

    if "reward" in metric:
        plt.axhline(
            0, color="black",
            linestyle="--", linewidth=1
        )

    plt.xlabel("Network Configuration (UAVs / IoT Nodes)")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs Network Configuration")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUT / filename, dpi=300)
    plt.close()

    print("Saved:", filename)


# Graph 10: Scalability of secure throughput
plot_metric(
    "secrecy_mbps_mean",
    "Mean Secrecy Rate (Mbps)",
    "10_network_vs_secrecy.png",
    ["mappo"]
)

# Graph 11: Scalability of energy efficiency
plot_metric(
    "secrecy_mbits_per_kj_mean",
    "Secure Energy Efficiency (Mbits/kJ)",
    "11_network_vs_efficiency.png",
    ["mappo"]
)

# Graph 12: UAV energy consumption
plot_metric(
    "uav_energy_j_per_step_mean",
    "Total UAV Energy (J/Step)",
    "12_network_vs_energy.png",
    ["mappo"]
)

# Graph 13: Evaluation reward
plot_metric(
    "reward_per_step_mean",
    "Mean Reward per Step",
    "13_network_vs_reward.png",
    ["mappo"]
)

# Graph 14: Battery depletion
plot_metric(
    "final_depleted_iot_fraction_mean",
    "Final Depleted IoT Fraction",
    "14_network_vs_depletion.png",
    ["mappo"]
)

# Graph 15: Comparison of three methods
plot_metric(
    "secrecy_mbps_mean",
    "Mean Secrecy Rate (Mbps)",
    "15_network_method_comparison.png",
    ["mappo", "random", "stationary"]
)

# Graph 16: Isolated IoT-size comparison
# UAV count is held fixed at 4.
subset = df[
    (df["policy"] == "mappo") &
    (df["num_uavs"] == 4)
].sort_values("num_iot")

if subset["num_iot"].nunique() >= 2:
    plt.figure(figsize=(8, 5))

    plt.plot(
        subset["num_iot"],
        subset["secrecy_mbps_mean"],
        marker="o",
        linewidth=2
    )

    plt.xlabel("Number of IoT Devices (4 UAVs Fixed)")
    plt.ylabel("Mean Secrecy Rate (Mbps)")
    plt.title("IoT Node Count vs Secrecy Rate")
    plt.xticks(sorted(subset["num_iot"].unique()))
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        OUT / "16_iot_count_vs_secrecy.png",
        dpi=300
    )
    plt.close()

    print("Saved: 16_iot_count_vs_secrecy.png")

print("\nAll supported scalability graphs generated.")
