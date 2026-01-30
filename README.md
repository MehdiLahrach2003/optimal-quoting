# Optimal Quoting
### Stochastic Control and Market Microstructure

Research-oriented Python sandbox for market making under microstructure uncertainty:
intensity calibration via maximum likelihood estimation, probing (controlled exploration),
and information–PnL / inventory-risk trade-offs.

This repository implements a toy yet research-structured framework to study how information
acquisition interacts with optimal quoting, parameter identifiability, and inventory risk.

---------------------------------------------------------------------

KEY RESULTS (VISUAL PREVIEW)

Information–PnL frontier under probing
Image file: reports/figures/frontier_pnl_vs_kerr.png

Intensity calibration via maximum likelihood estimation
Image file: reports/figures/intensity_fit.png

---------------------------------------------------------------------

QUICKSTART (2 MINUTES)

Clone the repository and install in editable mode:

git clone https://github.com/MehdiLahrach2003/Probing-and-adverse-selection-in-market-making.git
cd Probing-and-adverse-selection-in-market-making
pip install -e .

Run a minimal end-to-end smoke experiment:

python scripts/smoke.py

All results (CSV files and figures) are written automatically to:

reports/
reports/figures/

---------------------------------------------------------------------

WHY THIS PROJECT MATTERS

This project is designed as a research-grade sandbox for quantitative finance.

It demonstrates:
- how microstructure data limitations bias parameter estimation
- how controlled exploration (probing) restores identifiability
- how information acquisition impacts both PnL and inventory risk

The emphasis is not only on performance, but on understanding the structure
of the stochastic control problem.

---------------------------------------------------------------------

WHAT IS PROBING?

In microstructure models, key parameters such as trade intensities are often
poorly identified from passive market making data.

Probing refers to deliberately deviating from purely optimal quotes in order to:
- generate informative order flow
- reveal hidden adverse selection regimes
- improve calibration quality
- trade off short-term profitability for long-term information gain

This creates an explicit information–PnL frontier.

---------------------------------------------------------------------

CORE COMPONENTS

- Toy microstructure simulator with stochastic fills
- Intensity calibration via maximum likelihood estimation
- Quoting strategies:
  * baseline quoting
  * probing-enhanced quoting
  * Avellaneda–Stoikov policy
- Benchmarking framework:
  * PnL
  * volatility
  * inventory risk
- Frontier experiments:
  * probing parameter sweeps
  * information–PnL trade-offs

---------------------------------------------------------------------

MAIN EXPERIMENTS

Intensity calibration (MLE)
Script: scripts/calibrate_intensity.py
Output: reports/figures/intensity_fit.png

Avellaneda–Stoikov toy backtest
Script: scripts/run_as_toy.py
Output: reports/figures/as_equity.png

Baseline vs probing benchmark
Script: scripts/run_benchmark.py
Output: reports/benchmark_results.csv

Information–PnL frontier sweep
Script: scripts/run_frontier.py
Output: reports/figures/frontier_pnl_heatmap.png

Stress testing robustness
Script: scripts/run_stress.py
Output: reports/stress_results.csv

---------------------------------------------------------------------

REPRODUCIBILITY

The project is fully reproducible.

- experiments are executed via scripts located in the scripts/ directory
- experiment configurations are stored in the configs/ directory (YAML)
- generated outputs (CSV files and figures) are written to the reports/ directory

Results are intentionally not committed to version control to preserve
reproducibility and repository clarity.

---------------------------------------------------------------------

REPOSITORY STRUCTURE

src/optimal_quoting/
  backtest/        simulation engine
  calibration/     intensity estimation via MLE
  data/            dataset loading and schemas
  experiments/     probing frontier studies
  features/        microstructure feature builders
  metrics/         performance and inventory risk measures
  model/           intensity and Avellaneda–Stoikov components
  sim/             stochastic simulators
  strategy/        quoting policies (baseline, probing, A–S)
  config.py        configuration utilities
  log_utils.py     logging helpers
  types.py         typed structures

scripts/           reproducible experiment entry points
configs/           YAML experiment configurations
data/              raw and processed datasets (example included)
docs/              architecture, research notes, and roadmap
reports/           generated CSV files and figures
tests/             unit and integration test suite
.github/           CI workflow (pytest)

---------------------------------------------------------------------

SCOPE AND LIMITATIONS

This is a toy model, designed for clarity, experimentation, and research insight.
It is not intended for production trading.

Planned extensions include:
- transaction costs and adverse selection
- risk aversion tuning
- richer market impact models
- links to theoretical stochastic control results

---------------------------------------------------------------------

AUTHOR

Mehdi Lahrach
M1 Applied Mathematics and Statistics — Quantitative Finance
