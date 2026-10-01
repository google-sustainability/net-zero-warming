# Finite Amplitude Impulse Response (FaIR) Climate Model — Impact Replacement Approach

This repository contains the Python scripts and data required to replicate the
analysis and figures for our peer-reviewed submission. The codebase utilizes
the FaIR (Finite Amplitude Impulse Response) simple climate model to evaluate
the temperature impacts of mixed mitigation portfolios, combining Short-Lived
Climate Pollutant (SLCP) mitigation with long-term Carbon Dioxide Removal (CDR).

## Repository Structure

* `Inputs/`: Directory containing necessary FaIR calibration files (e.g.,
  constrained parameters, species configs, and emissions data).
* `Outputs/`: Directory where all generated figures (PNG) and intermediate
  data (CSV) are saved.
* `fair_utils.py`: Core utility functions, FaIR model wrappers, optimization
  solvers, and global constants.
* `generate_temperature_responses_*.py`: Scripts to pre-calculate temperature
  response curves for CO₂, CH₄, and HFC-134a. Note that these are NOT used
  in the models presented in the manuscript, but are included for completeness
  here as they enable quick execution of solver routines.
* `run_calc_min_*.py`: Solvers that calculate the minimum mitigation required
  to neutralize given pulses. Also generate Figures 2(a) and (b) in the
  manuscript.
* `generate_figure_*.py`: Scripts that generate the final figures presented
  in the manuscript.

## Prerequisites

Ensure you have Python 3.8+ installed. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Running the Code

### Generate Final Manuscript Figures and Tables

These scripts run the iterative solvers to balance emissions with mitigation
and produce Figure 2(a) and 2(b):

```bash
python run_calc_min_cdr.py
python run_calc_min_ch4.py
# Note: run_calc_min_hfc.py is NOT used to produce a figure in the manuscript;
# included for completeness.
python run_calc_min_hfc.py
```

These scripts generate the remaining figures and tables:

```bash
python generate_figure_1.py
python generate_figure_3.py
python generate_figure_4.py
python run_calc_min_mitigation_delay.py
python run_calc_min_mitigation_delay_forcing.py
```

## Vulnerability Reporting

Eligibility for the
[Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security)
is determined by the
[Google Open Source Software Vulnerability Reward Program Rules](https://bughunters.google.com/about/rules/open-source/google-open-source-software-vulnerability-reward-program-rules).