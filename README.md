# Sustained Neutralization of the Warming Response to CO₂ Emissions

This repository contains the Python code and configuration files required to
reproduce the analyses, tables, and figures for *Sustained Neutralization of
the Warming Response to CO₂ Emissions through Multiple Mitigation Strategies*.

The codebase uses the Finite Amplitude Impulse Response (FaIR v2.2)
reduced-complexity climate model to evaluate the continuous global mean surface
temperature (GMST) response to residual CO₂ emissions compensated through
portfolios of short-lived climate pollutant (SLCP) abatement,
shorter-durability carbon dioxide removal (CDR), and durable CDR under the
warming neutralization approach ('net-zero warming').

## Repository Structure

* `Inputs/`: Directory for FaIR calibration, exogenous forcing, and baseline
  emissions files (see **Input Data and Licensing** below).
* `Outputs/`: Directory where generated figures (PNG) and intermediate summary
  and trajectory tables (CSV) are saved.
* `Master.ipynb`: Jupyter notebook orchestrating end-to-end figure and table
  generation.
* `fair_utils.py`: Core FaIR v2.2 model wrapper (`run_fair`), quantile export
  utilities, iterative binary search solvers for stacking (`calc_min_cdr`),
  bridging (`calc_min_ch4`, `calc_min_hfc`), and standardized replacement
  horizons (`calc_min_mitigation_delay`), and global physical/GWP constants.
* `generate_temperature_responses_*.py`: Scripts to pre-calculate unit impulse
  temperature response curves (`co2`, `ch4`, `hfc`). While manuscript results
  use the full iterative binary search solver (`use_solver=True` in
  `fair_utils.py`) rather than linear impulse superposition, these response
  curves seed initial search bounds for `calc_min_cdr`, `calc_min_ch4`, and
  `calc_min_hfc`.
* `run_calc_min_cdr.py`, `run_calc_min_ch4.py`, `run_calc_min_hfc.py`: Solvers
  that calculate minimum progressive durable CDR additions (stacking; Figure
  2a) or SLCP mitigation additions across ramped delayed CDR (bridging; Figure
  2b). (`run_calc_min_hfc.py` is included for completeness alongside CH₄.)
* `run_calc_min_mitigation_delay.py`: Calculates conservative (p95)
  standardized relationships between CO₂ emissions, upfront SLCP mitigation,
  and durable CDR replacement horizons from 1 to 20 years (Table 2 and
  Supplementary Tables S1 and S2).
* `run_calc_min_mitigation_delay_forcing.py`: Implements the radiative forcing
  (RF) neutralization comparison in FaIR across 5- to 20-year replacement
  horizons (Supplementary Section S1 and Supplementary Table S4).
* `generate_figure_1.py`: Generates Figure 1 (modeled GMST response to a 1 Gt
  CO₂ emission pulse and 1 Gt CO₂e (GWP20) CH₄ mitigation pulse in 2030).
* `generate_figure_3.py`: Generates Figure 3 (application of the standardized
  relationships in Table 2 to neutralize a multi-year CO₂ emission profile
  using CH₄ mitigation, HFC-134a mitigation, and durable CDR).
* `run_calibration_sensitivity.py`: Quantifies sensitivity of standardized CH₄
  mitigation ratios across 5- to 20-year horizons under four FaIR
  configurations: baseline `fair-calibrate` v1.4.1 with v1.4.0 literature CH₄
  lifetime parameters, unmodified v1.4.1, CMIP7/IGCC-constrained v1.6.0, and a
  15% shorter unperturbed CH₄ lifetime (Supplementary Table S5).
* `run_midcourse_reassessment.py`: Simulates dynamic portfolio management and
  generates Figure 4, showing portfolio adaptation to (a) a 5-year durable CDR
  delivery delay (2042 to 2047) and (b) a 2037 revision reducing unperturbed
  CH₄ lifetime by 15%.

## Prerequisites

Ensure you have Python 3.9+ installed. Install required dependencies (including
`fair`, `numpy`, `pandas`, `xarray`, `matplotlib`, and `pooch`) via:

```bash
pip install -r requirements.txt
```

## Input Data and Licensing

### 1. ScenarioMIP Baseline Emissions (User-Supplied)

Baseline scenario emissions data (`EMISSIONS_FILE = "extensions_1750-2500.csv"`
in `fair_utils.py`) are derived from ScenarioMIP and **cannot be re-hosted or
redistributed in this repository** under the ScenarioMIP Data License.

Before running the code:
1. Access the ScenarioMIP dataset directly via the IIASA ScenarioMIP Scenario
   Explorer: https://scenariomip.apps.ece.iiasa.ac.at/
2. Place the FaIR-formatted scenario emissions CSV in `Inputs/` as:
   `Inputs/extensions_1750-2500.csv` (configured in `fair_utils.py` via
   `DATA_DIR`, `EMISSIONS_FILE`, and `SCENARIO = "low"`).

### 2. FaIR Calibration and Custom Baseline Species Files (CC BY 4.0)

Calibration and exogenous forcing files in `Inputs/` are adapted from
`fair-calibrate` under the Creative Commons Attribution 4.0 International
(CC BY 4.0) license:

* `Inputs/calibrated_constrained_parameters_calibration1.4.1.csv` and
  `Inputs/volcanic_solar.csv`: Sourced from `fair-calibrate` v1.4.1 (Smith et
  al., 2024; https://doi.org/10.5281/zenodo.10566813).
* `Inputs/species_configs_properties_calibration1.4.1_ch4modtemp.csv`: Custom
  baseline species configuration provided in this repository, combining
  `fair-calibrate` v1.4.1 species properties with `fair-calibrate` v1.4.0
  literature-based CH₄ unperturbed lifetime, temperature sensitivity, and
  chemical sensitivities (https://doi.org/10.5281/zenodo.8399112; see Methods).
* **Alternative calibrations (auto-downloaded):** When running
  `run_calibration_sensitivity.py` or `run_midcourse_reassessment.py`,
  unmodified `fair-calibrate` v1.4.1 (`10.5281/zenodo.10566813`) and
  CMIP7/IGCC-constrained v1.6.0 (`fastmip v1`; Smith, 2026;
  https://doi.org/10.5281/zenodo.18828694) files are downloaded automatically
  from Zenodo into `Inputs/` via `pooch` if not already present.

## Running the Code

You can execute the full replication workflow interactively via `Master.ipynb`,
or run individual scripts from the command line once
`Inputs/extensions_1750-2500.csv` is in place.

1. **(Optional, if `Outputs/` is empty) Pre-calculate unit impulse response
   curves** used to seed solver search bounds in `fair_utils.py`:

```bash
python generate_temperature_responses_co2.py
python generate_temperature_responses_ch4.py
python generate_temperature_responses_hfc.py
```

2. **Run stacking and bridging optimization solvers** (produces Figures 2a and
   2b when called with manuscript scenario inputs in `Master.ipynb`):

```bash
python run_calc_min_cdr.py
python run_calc_min_ch4.py
python run_calc_min_hfc.py
```

3. **Generate standardized lookup tables, sensitivity analyses, and remaining
   manuscript figures:**

```bash
python generate_figure_1.py
python generate_figure_3.py
python run_calc_min_mitigation_delay.py
python run_calc_min_mitigation_delay_forcing.py
python run_calibration_sensitivity.py
python run_midcourse_reassessment.py
```

## Vulnerability Reporting

Eligibility for the
[Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security)
is determined by the
[Google Open Source Software Vulnerability Reward Program Rules](https://bughunters.google.com/about/rules/open-source/google-open-source-software-vulnerability-reward-program-rules).
