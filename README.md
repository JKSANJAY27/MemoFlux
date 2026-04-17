# AX Memory — Context-Aware Adaptive Memory for Samsung Mobile Agents

> **Samsung AX Hackathon 2026 | Problem Statement 03**
> Built by JKSANJAY27 | Week 1 Submission

---

## Problem Statement

Modern Android devices manage memory reactively — apps are killed and cold-started based on simple LRU heuristics that were designed in the pre-AI era. As Samsung Galaxy devices become AI-first (Galaxy AI, on-device inference, Bixby agents), the memory subsystem becomes the bottleneck. A 900ms cold start for Chrome isn't just annoying — it breaks the seamless flow that Samsung AX promises.

**AX Memory** replaces dumb LRU with a context-aware, predictive memory manager that pre-loads apps before the user opens them, learns their daily patterns, and runs entirely on-device — no cloud dependency, full privacy.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download LSApp dataset and setup data directories
python setup_data.py

# 3. Run baseline benchmarks (LRU / LFU / Static Priority)
python -m evaluation.benchmark_runner

# 4. Launch the interactive dashboard
streamlit run dashboard/app.py

# 5. Generate synthetic stress-test data
python data_pipeline/synthetic_generator.py --n_users 100
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     AX Memory — System Architecture                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  REAL DATA               DATA PIPELINE              SIMULATION          │
│  ──────────             ─────────────────          ────────────         │
│  LSApp TSV    ──────►   lsapp_loader.py   ──────►  MemorySimEnv         │
│  (599K events)          feature_engineer            (Gymnasium)          │
│                         session_builder                                  │
│                                                        │                │
│  SYNTHETIC              ─────────────────              │                │
│  Generator    ──────►   synthetic_gen.py   ──────────►─┤                │
│  (Weibull/Poisson)                                     │                │
│                                                        ▼                │
│  MEMORY MANAGERS        EVALUATION ENGINE         KPI TRACKER           │
│  ─────────────         ─────────────────          ───────────           │
│  LRU (baseline) ──────►  benchmark_runner  ──────► 7 KPIs               │
│  LFU (baseline)         report_generator           Dashboard            │
│  Static Priority                                                         │
│  [Week 2] LSTM Predictor                                                │
│  [Week 3] RL Agent (PPO)                                                │
│                                                                         │
│  DASHBOARD                                                              │
│  ──────────────────────────────────────────────────────────             │
│  Tab 1: Live Demo     Tab 2: Baseline vs AX     Tab 3: Dataset          │
│  (Memory Map Grid)    (KPI Comparison)          (LSApp Explorer)        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Datasets

### LSApp (Primary — Required)
- **Source**: https://github.com/aliannejadi/LSApp
- **Size**: 599,635 app usage records, 292 users, 8 months
- **Download**: Auto-downloaded by `python setup_data.py`
- **Citation**: Aliannejadi et al., ACM TOIS 2021

### Kaggle Mobile Device Usage (Supplementary)
- **Source**: https://www.kaggle.com/datasets/valakhorasani/mobile-device-usage-and-user-behavior-dataset
- **Use**: User archetype labeling (light/heavy user segmentation)
- **Download**: `kaggle datasets download valakhorasani/mobile-device-usage-and-user-behavior-dataset`

### Synthetic Data (Always Available)
```bash
python data_pipeline/synthetic_generator.py --n_users 500 --days 30
```

---

## KPI Targets (PS-03 Evaluation Criteria)

| # | KPI | Baseline (LRU) | Week 3 Target | Method |
|---|-----|---------------|---------------|--------|
| 1 | Avg App Load Time | ~228 ms | ≤182 ms (↓20%) | Env timing |
| 2 | Cold Start Rate | ~35% | ≤25% (↓28%) | Cache tracking |
| 3 | Memory Thrashing | ~18/100 steps | ≤9/100 (↓50%) | Env events |
| 4 | System Stability | 99.4% | 100% | OOM tracking |
| 5 | Next-App HR@3 | 0% (reactive) | ≥75% | LSTM predictor |
| 6 | Cache Hit Rate | ~65% | ≥85% | Cache tracking |
| 7 | Memory Efficiency | ~42% | ≥72% (↓30%) | Look-ahead |

---

## Week-by-Week Roadmap

| Week | Deliverable | Status |
|------|-------------|--------|
| **Week 1** | Gymnasium env, data pipeline, LRU/LFU/Static baselines, Streamlit dashboard, unit tests | ✅ **Complete** |
| **Week 2** | LSTM next-app predictor (PyTorch), trained on LSApp, achieves HR@3 ≥ 75% | 🔜 Planned |
| **Week 3** | PPO RL agent (Stable-Baselines3) trained in MemorySimEnv, beats all baselines | 🔜 Planned |
| **Week 4** | On-device integration demo, explainability layer, final eval report | 🔜 Planned |

---

## Team

- **JKSANJAY27** — Lead Engineer

---

## Samsung AX Connection

This system is designed for on-device deployment on Samsung Galaxy devices:
- **Galaxy S24** (8GB RAM, 10 memory slots) — primary target
- **Galaxy A54** (6GB RAM, 8 slots) — mid-range target
- **Galaxy A34** (4GB RAM, 6 slots) — budget target

The memory manager runs entirely on-device as a Samsung AX service, requiring no cloud connectivity. All inference is performed locally using Samsung's NPU (via PyTorch/ExecuTorch). User data never leaves the device — addressing privacy concerns that cloud-based approaches cannot.

---

## Project Structure

```
ax-memory/
├── README.md
├── requirements.txt
├── setup_data.py            # Dataset download + setup
├── demo_scenario.py         # 10-step compelling demo
├── data/
│   ├── raw/                 # LSApp TSV (downloaded)
│   ├── processed/           # Feature-engineered Parquet/pickle
│   └── synthetic/           # Generated sessions
├── env/
│   ├── memory_sim_env.py    # Gymnasium MemorySimEnv (CORE)
│   ├── app_registry.py      # App metadata
│   └── device_profiles.py   # Samsung device configs
├── data_pipeline/
│   ├── lsapp_loader.py      # LSApp parser
│   ├── feature_engineer.py  # 12-dim context vector
│   ├── session_builder.py   # Episode constructor
│   └── synthetic_generator.py
├── baselines/
│   ├── lru_manager.py       # LRU (naive phone)
│   ├── lfu_manager.py       # LFU
│   └── static_priority.py   # Frequency-ranked static
├── evaluation/
│   ├── kpi_tracker.py       # 7 PS-03 KPIs
│   ├── benchmark_runner.py  # 500-episode runner
│   └── report_generator.py  # JSON + Markdown reports
├── dashboard/
│   ├── app.py               # Streamlit (3 tabs)
│   └── components/
└── tests/
    ├── test_env.py           # 10 required tests
    ├── test_data_pipeline.py
    └── test_baselines.py
```

---

*AX Memory is open-sourced for the Samsung AX Hackathon 2026. All Samsung Galaxy device profiles, app load time measurements, and memory footprint data are based on publicly available Android benchmarks.*
