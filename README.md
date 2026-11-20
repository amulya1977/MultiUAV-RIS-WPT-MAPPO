# Multi-UAV RIS-WPT-MAPPO

**Multi-Agent Reinforcement Learning for RIS-Assisted Secure UAV Communications and Wireless Power Transfer**

A Python simulation and research codebase for studying **physical-layer security (PLS)** in multi-UAV networks assisted by a **reconfigurable intelligent surface (RIS)**, with **wireless power transfer (WPT)** to IoT devices and **multi-agent proximal policy optimization (MAPPO)** for control. The main research implementation is in [`pls_mappo/`](pls_mappo/).

> **Scope clarification:** This repository also retains an earlier, separate diffusion-enhanced multi-UAV **mobile-edge-computing (MEC)** framework (`src/`, root `train.py`, `test_custom.py`). That code optimizes task offloading, caching, latency and energy; **it is not the RIS/WPT/PLS experiment environment**. The PLS MAPPO actor in `pls_mappo/train_mappo.py` is a **Gaussian policy, not a diffusion actor**. Results and comparisons from the two pipelines must not be mixed.

## Overview

The PLS simulation combines multiple UAVs, IoT devices, passive eavesdroppers and a configurable RIS. UAV actions affect movement, WPT/communication decisions, artificial noise and RIS phase settings. The environment computes legitimate and eavesdropper link behavior, secrecy rate, UAV propulsion/operating energy, IoT energy harvesting and battery status, and safety-related motion metrics. MAPPO learns decentralized actions with a shared actor and a centralized critic during training (CTDE).

### Research objectives

- Investigate secure UAV-to-IoT communication in the presence of eavesdroppers.
- Study RIS phase-control effects on modeled wireless channels.
- Track the trade-off between **secrecy throughput**, **UAV energy expenditure**, and **IoT battery/harvesting outcomes**.
- Train and evaluate a multi-UAV MAPPO policy against reproducible random and stationary control baselines.
- Analyze sensitivity, scalability, communication-load settings, and evaluation reliability using recorded experimental outputs.

## System workflow

```text
IoT devices + eavesdroppers + RIS + multiple UAVs
                         |
             PLSMultiUAVEnv (simulator)
                         |
              Local observations per UAV
                         |
                Shared Gaussian actor
                         |
     UAV motion / WPT / artificial noise / RIS decisions
                         |
     Channel + secrecy + energy + battery calculations
                         |
             Rewards and global state
                         |
        Centralized critic + PPO updates (MAPPO)
                         |
      Saved checkpoints, metrics and evaluations
```

## What is implemented

| Component | Location | Purpose |
|---|---|---|
| Scenario parameters | `pls_mappo/config.py` | Reproducible UAV/IoT/RIS deployment defaults |
| PLS environment | `pls_mappo/environment.py` | Integrated multi-UAV step/reset API, rewards, safety and battery observations |
| RIS/channel calculations | `pls_mappo/channel_model.py` | Modeled direct/RIS-assisted propagation |
| UAV/IoT energy | `pls_mappo/energy_model.py` | UAV operating energy, wireless charging and IoT battery updates |
| Secrecy model | `pls_mappo/security_model.py` | Legitimate/eavesdropper rate and secrecy-rate calculations |
| MAPPO trainer | `pls_mappo/train_mappo.py` | Shared Gaussian actor, central critic, PPO training, checkpointing |
| Checkpoint evaluation | `pls_mappo/evaluate_mappo.py` | Held-out seeded comparison against random and stationary baselines |
| Research experiment runner | `run_research_experiments.py` | Sensitivity, scalability and transmit-power-demand experiment sweeps |
| Gap experiments | `run_gap_experiments.py` | Targeted UAV-count and communication-load sweeps |
| Plot scripts | `generate_research_graphs.py`, `generate_sensitivity_graphs.py`, `generate_scalability_graphs.py`, `generate_reliability_graphs.py`, `plot_gap_results.py` | Generate visualizations from supported experiment files |
| Diagnostic tests | `test_channels.py`, `test_energy.py`, `test_security.py`, `test_environment.py`, `test_stage7b.py`, `test_stage7c.py`, `test_stage7f.py` | Component and integration checks |

The root also contains `diagnose_wpt.py`, `diagnose_channel_7e.py`, and `diagnose_battery_stage7g.py` for focused debugging.

## Default PLS scenario

Illustrative defaults from `pls_mappo/config.py`:

| Parameter | Default |
|---|---:|
| UAVs | 3 |
| IoT devices | 12 |
| Eavesdroppers | 2 |
| RIS elements | 32 |
| Maximum configured UAVs | 8 |
| Deployment area side length | 200 m |
| Fixed UAV altitude | 40 m |
| Time-slot duration | 1 s |
| Maximum UAV speed | 10 m/s |
| Minimum UAV separation | 5 m |
| Random seed | 42 |

These are **simulation defaults**, not measurements calibrated to a physical deployment.

## Installation

Use Python 3.10 or newer. Install the dependencies declared by the repository:

```bash
git clone https://github.com/amulya1977/MultiUAV-RIS-WPT-MAPPO.git
cd MultiUAV-RIS-WPT-MAPPO
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Then install:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements include NumPy, PyTorch, Matplotlib, ImageIO and ImageIO-FFmpeg. The PLS trainer supports CPU and can use CUDA when a compatible PyTorch installation is available.

## Quick start: RIS-WPT-PLS project

Run component checks from the **repository root**:

```bash
python test_channels.py
python test_energy.py
python test_security.py
python test_environment.py
```

Run a **short pipeline smoke test** (checks that training runs; not evidence of policy convergence):

```bash
python -m pls_mappo.train_mappo --updates 2 --rollout-steps 16 --device cpu
```

Train a larger experimental run, adjusting the budget to your compute resources:

```bash
python -m pls_mappo.train_mappo \
  --updates 100 \
  --rollout-steps 256 \
  --episode-steps 100 \
  --num-uavs 3 \
  --num-iot 12 \
  --actor-lr 0.00003 \
  --critic-lr 0.0001 \
  --minibatch-size 256 \
  --seed 42 \
  --save-every 25 \
  --output-dir pls_mappo/results
```

For Windows PowerShell, enter the same command on one line or replace shell line continuations with PowerShell backticks.

Training writes `run_config.json`, `training_metrics.json`, and model checkpoints in the selected output directory. In particular, the trainer writes `mappo_latest.pt` and periodic `mappo_update_XXXXX.pt` files.

## Evaluating a trained policy

The evaluation script requires an existing checkpoint. The following works **after the above training run**:

```bash
python -m pls_mappo.evaluate_mappo \
  --checkpoints pls_mappo/results/mappo_latest.pt \
  --episodes 10 \
  --episode-steps 100 \
  --num-uavs 3 \
  --num-iot 12 \
  --output-dir pls_mappo/results_stage7i_eval
```

The evaluation compares the checkpoint's deterministic actor-mean actions against **random** and **stationary** policies under matched evaluation seed indexes. It writes:

- `episodes.csv` — per-episode outcomes;
- `summary.json` — mean and standard deviation by policy;
- `evaluation_config.json` — parameters and evaluation notes;
- comparison plots for secrecy, energy, efficiency, final battery and rejected UAV moves.

**Important:** A checkpoint must match the evaluation environment's observation, action, state and agent dimensions. A trained policy may not outperform baselines in every scenario.

## Metrics

The PLS pipeline records, among other quantities:

| Metric | Interpretation |
|---|---|
| Secrecy rate (Mbps) | Modeled secure communication throughput |
| UAV energy (J/step) | Aggregate UAV energy per simulation step |
| Secrecy efficiency (Mbits/kJ) | Secure data per unit UAV energy |
| Harvested energy (J/step) | Wireless energy received by IoT devices |
| IoT transmission energy (J/step) | IoT energy spent on uplink transmission |
| IoT battery level / depleted fraction | Device energy sustainability indicators |
| Collision reversions (per step) | Movement attempts rejected by separation constraints |
| Flight distance (m/step) | Actual and attempted UAV motion |
| Active IoT devices | Number of communicating/active devices |

Training metrics are **not automatically generalization results**; use held-out evaluation, multiple training seeds, consistent budgets and uncertainty summaries for comparative conclusions.

## Research experiments

Run from the repository root:

```bash
# Small verification of the sensitivity runner (not a publication-quality experiment)
python run_research_experiments.py --mode sensitivity --updates 2 --max-cases 1

# Scalability sweep with multiple independent seeds and more evaluation episodes
python run_research_experiments.py --mode scalability --updates 100 --eval-episodes 10 --train-seeds 42 43 44

# IoT transmit-power demand sweep
python run_research_experiments.py --mode traffic --updates 100 --eval-episodes 10 --train-seeds 42 43 44

# Targeted missing-case sweeps
python run_gap_experiments.py --gap uavs --updates 100
python run_gap_experiments.py --gap traffic --updates 100
```

The runner supports `sensitivity`, `scalability`, and `traffic`; `traffic` means a **transmit-power-demand proxy**, not MEC computational task size. Run `python run_research_experiments.py --help` for all supported arguments. Successful experiments record case configurations, checkpoint files, CSV summaries/episodes and training logs; failures may be recorded in `failure.txt`.

### Experimental validity

The research runner's defaults (`--updates 2`, `--eval-episodes 2`) are intended for a **smoke test**. Larger training budgets, held-out seeds, repeated independent trials and a documented evaluation protocol are needed before reporting stable scientific comparisons. Do not treat any single run as proof of general superiority.

## Results availability

**There is no external Google Drive results link in this README.** This repository contains experiment scripts and research folders, but the presence of a script does not establish that every figure, checkpoint, CSV or claimed comparison has been produced and published.

To inspect available outputs, browse [`pls_mappo/`](pls_mappo/) and generated results directories in your local clone. Git may omit large outputs depending on `.gitignore`, so locally generated figures and checkpoints might not appear on GitHub. Run training/evaluation to regenerate outputs where feasible. Numerical results should be reported with the exact checkpoint, configuration and dataset/seed information that produced them.

## Modeling assumptions and limitations

- The PLS environment is a **research simulation**, not a validated real-world UAV/RIS testbed.
- UAV altitude is fixed in the current PLS environment; an altitude-related action is reserved/ignored.
- Channels include simulated stochastic fading; results depend on model assumptions and seeds.
- RIS elements are partitioned using the configured maximum UAV count.
- QoS and depot-return conditions are diagnostic rather than fully enforced as hard constraints in the current environment.
- Rewards use normalized quantities; a reward score should not be interpreted as a physical-unit measurement.
- Wireless charging is modeled, but positive harvesting alone does **not** demonstrate indefinite battery sustainability.
- The PLS learner is **Gaussian MAPPO**, not the diffusion-enhanced agent from the older MEC code.

## Legacy MEC / diffusion-MAPPO implementation

The repository also includes a distinct UAV-assisted mobile-edge-computing implementation:

- `src/` — MEC environment, diffusion policy, MAPPO and comparison baselines;
- `train.py` — legacy MEC training (`DMJO`, `MAPPO`, `MADDPG`, `MATD3`, `GreedyOffloading`, `LocalExecution`);
- `test_custom.py` — MEC scenario evaluation;
- `plot_ieee.py` — MEC-focused plotting.

For that **separate pipeline**:

```bash
python train.py --method DMJO --total-steps 1000 --device cpu --smoke-test --no-training-video
```

This command is an illustrative small run for the older MEC system, **not** the RIS-WPT-PLS research implementation. Avoid citing MEC latency/cache metrics as PLS secrecy/WPT results.

## Reproducibility checklist

When presenting results in a report or paper, retain the source revision, Python/PyTorch versions, command-line arguments, configuration, number of training updates, rollout length, evaluation seed range, number of evaluation episodes, raw CSVs, failures and plotting scripts. Mark missing or failed runs instead of replacing them with estimated values.

## Repository

Maintainer: [amulya1977](https://github.com/amulya1977)  
Source: [MultiUAV-RIS-WPT-MAPPO](https://github.com/amulya1977/MultiUAV-RIS-WPT-MAPPO)

**License:** No license grant is claimed here; check for an explicit `LICENSE` file before assuming permission to reuse or redistribute the code.

