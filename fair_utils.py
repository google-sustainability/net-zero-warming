#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions and core routines for running the FaIR climate model."""

import os
from fair import FAIR
from fair import interface
from fair.io import read_properties
import numpy as np
import pandas as pd
import xarray as xr

# ==========================================
# Global Configurations
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Inputs")
TARGET_DIR = os.path.join(BASE_DIR, "Outputs")

PARAMS_FILE = os.path.join(
    DATA_DIR, "calibrated_constrained_parameters_calibration1.4.1.csv"
)
SPECIES_FILE = os.path.join(
    DATA_DIR, "species_configs_properties_calibration1.4.1_ch4modtemp.csv"
)
EMISSIONS_FILE = "extensions_1750-2500.csv"
FORCING_FILE = "volcanic_solar.csv"

SCENARIO = "low"
MIN_YEAR = 1750
MAX_YEAR = 2500
START_YEAR = 2024
TARGET_YEAR = 2030
OFFSET = TARGET_YEAR - START_YEAR

USE_THORNHILL = False
HFC_SPECIE = "HFC-134a"
EVAL_PERCENTILE = 0.95
TOLERANCE = 0.01
GWP100_CH4 = 27.9
GWP20_CH4 = 81.2
GWP100_HFC = 1530
GWP20_HFC = 4140


# ==========================================
# Core Functions
# ==========================================
def run_fair(co2_adj=None, ch4_adj=None, hfc_adj=None, hfc_specie=HFC_SPECIE):
  """Runs the FaIR model with specified emissions adjustments.

  Args:
    co2_adj: Array of CO2 emissions adjustments.
    ch4_adj: Array of CH4 emissions adjustments.
    hfc_adj: Array of HFC emissions adjustments.
    hfc_specie: Name of the HFC species to adjust.

  Returns:
    The configured and executed FAIR model instance.
  """
  if co2_adj is None:
    co2_adj = np.zeros(1000)
  if ch4_adj is None:
    ch4_adj = np.zeros(1000)
  if hfc_adj is None:
    hfc_adj = np.zeros(1000)

  if USE_THORNHILL:
    f = FAIR(ch4_method="Thornhill2021")
  else:
    f = FAIR()

  f.define_time(MIN_YEAR, MAX_YEAR, 1)
  f.define_scenarios([SCENARIO])

  df_configs = pd.read_csv(PARAMS_FILE, index_col=0)
  f.define_configs(df_configs.index)
  species, properties = read_properties(filename=SPECIES_FILE)
  f.define_species(species, properties)
  f.allocate()

  f.fill_from_csv(
      emissions_file=os.path.join(DATA_DIR, EMISSIONS_FILE),
      forcing_file=os.path.join(DATA_DIR, FORCING_FILE),
  )

  time_mask = f.emissions["timepoints"] >= START_YEAR
  time_sel = f.emissions["timepoints"].where(time_mask, drop=True)
  n_years = time_sel.size

  adj_da_co2 = xr.DataArray(
      co2_adj[:n_years], coords={"timepoints": time_sel}, dims="timepoints"
  )
  co2_slice = dict(timepoints=time_sel, specie="CO2 FFI")
  f.emissions.loc[co2_slice] = f.emissions.sel(**co2_slice) + adj_da_co2

  adj_da_ch4 = xr.DataArray(
      ch4_adj[:n_years], coords={"timepoints": time_sel}, dims="timepoints"
  )
  ch4_slice = dict(timepoints=time_sel, specie="CH4")
  f.emissions.loc[ch4_slice] = f.emissions.sel(**ch4_slice) + adj_da_ch4

  adj_da_hfc = xr.DataArray(
      hfc_adj[:n_years], coords={"timepoints": time_sel}, dims="timepoints"
  )
  hfc_slice = dict(timepoints=time_sel, specie=hfc_specie)
  f.emissions.loc[hfc_slice] = f.emissions.sel(**hfc_slice) + adj_da_hfc

  interface.fill(
      f.forcing,
      f.forcing.sel(specie="Volcanic")
      * df_configs["forcing_scale[Volcanic]"].values.squeeze(),
      specie="Volcanic",
  )
  interface.fill(
      f.forcing,
      f.forcing.sel(specie="Solar")
      * df_configs["forcing_scale[Solar]"].values.squeeze(),
      specie="Solar",
  )
  f.fill_species_configs(SPECIES_FILE)
  f.override_defaults(PARAMS_FILE)

  interface.initialise(
      f.concentration, f.species_configs["baseline_concentration"]
  )
  interface.initialise(f.forcing, 0)
  interface.initialise(f.temperature, 0)
  interface.initialise(f.cumulative_emissions, 0)
  interface.initialise(f.airborne_emissions, 0)
  interface.initialise(f.ocean_heat_content_change, 0)

  f.run()
  return f


def export_diff_scenarios(
    f_base, co2_adj=None, ch4_adj=None, hfc_adj=None, savename="intervention_"
):
  """Exports temperature and forcing difference scenarios to CSV files.

  Args:
    f_base: Baseline FAIR model instance.
    co2_adj: Array of CO2 emissions adjustments.
    ch4_adj: Array of CH4 emissions adjustments.
    hfc_adj: Array of HFC emissions adjustments.
    savename: Prefix for output CSV filenames.
  """
  print(f"Running adjusted scenario for export: {savename}_{SCENARIO}...")
  f_adj = run_fair(co2_adj=co2_adj, ch4_adj=ch4_adj, hfc_adj=hfc_adj)

  t_base = f_base.temperature.sel(scenario=SCENARIO, layer=0).squeeze(drop=True)
  t_adj = f_adj.temperature.sel(scenario=SCENARIO, layer=0).squeeze(drop=True)
  temp_diff = t_adj - t_base

  f_base_forcing = f_base.forcing_sum.sel(scenario=SCENARIO).squeeze(drop=True)
  f_adj_forcing = f_adj.forcing_sum.sel(scenario=SCENARIO).squeeze(drop=True)
  forcing_diff = f_adj_forcing - f_base_forcing

  def format_quantiles(diff_data):
    percentiles_to_calc = list(set([0.05, 0.50, 0.95, EVAL_PERCENTILE]))
    percentiles_to_calc.sort()

    q = diff_data.quantile(percentiles_to_calc, dim="config")
    time_dim = [d for d in q.dims if d != "quantile"][0]
    df = q.transpose(time_dim, "quantile").to_pandas()
    df.columns = [f"p{int(col * 100):02d}" for col in df.columns]

    # Reset index to push the native FaIR time dimension into column 0
    return df.reset_index()

  df_temp = format_quantiles(temp_diff)
  temp_csv_path = os.path.join(
      TARGET_DIR, f"temperature_{savename}_{SCENARIO}.csv"
  )
  df_temp.to_csv(temp_csv_path, index=False)

  df_forcing = format_quantiles(forcing_diff)
  forcing_csv_path = os.path.join(
      TARGET_DIR, f"forcing_{savename}_{SCENARIO}.csv"
  )
  df_forcing.to_csv(forcing_csv_path, index=False)

  print(f"Export complete. Saved to: {TARGET_DIR}")


def calc_min_mitigation_delay(
    delay_years, t_base, target_gas="CH4", search_low=0.0, search_high=50.0
):
  """Calculates the minimum SLCP mitigation required for a given delay.

  Args:
    delay_years: Number of years to delay the mitigation.
    t_base: Baseline temperature data array.
    target_gas: Target greenhouse gas ('CH4' or 'HFC').
    search_low: Lower bound for bisection search.
    search_high: Upper bound for bisection search.

  Returns:
    A tuple of (mitigation_amount, iterations).
  """
  idx_start_eval = (TARGET_YEAR + 1) - MIN_YEAR
  idx_end_eval = (TARGET_YEAR + delay_years) - MIN_YEAR

  low, high = search_low, search_high
  iteration = 0

  while (high - low) > TOLERANCE:
    iteration += 1
    mid_val = (low + high) / 2.0

    co2_pulse = np.zeros(1000)
    co2_pulse[OFFSET] = 1.0
    co2_pulse[OFFSET + delay_years] = -1.0

    mitigation_pulse = np.zeros(1000)
    mitigation_pulse[OFFSET] = -mid_val

    if target_gas.upper() == "CH4":
      f_scenario = run_fair(co2_adj=co2_pulse, ch4_adj=mitigation_pulse)
    elif target_gas.upper() == "HFC":
      f_scenario = run_fair(co2_adj=co2_pulse, hfc_adj=mitigation_pulse)
    else:
      raise ValueError("target_gas must be either 'CH4' or 'HFC'")

    t_scenario = f_scenario.temperature.sel(scenario=SCENARIO, layer=0).squeeze(
        drop=True
    )
    temp_diff = (t_scenario - t_base).quantile(EVAL_PERCENTILE, dim="config")
    max_t_anomaly = temp_diff[idx_start_eval : idx_end_eval + 1].max().values

    if max_t_anomaly > 0:
      low = mid_val
    else:
      high = mid_val

  return high, iteration


def calc_min_cdr(
    co2_emission_adj=None,
    co2_cdr_adj=None,
    ch4_adj=None,
    hfc_adj=None,
    use_solver=True,
    tolerance=TOLERANCE,
):
  """Calculates the minimum CDR required to neutralize a CO2 pulse.

  Includes an optional iterative solver to account for carbon cycle asymmetries.
  Emissions can include CO2 only. Existing mitigation can include CDR, CH4 and
  HFC.

  Args:
    co2_emission_adj: Array of CO2 emission adjustments.
    co2_cdr_adj: Array of CO2 CDR adjustments.
    ch4_adj: Array of CH4 adjustments.
    hfc_adj: Array of HFC adjustments.
    use_solver: Whether to use the iterative solver.
    tolerance: Convergence tolerance for the solver.

  Returns:
    A tuple of (additional_cdr_required, df_uncomp, df_comp).
  """
  response_percentile = EVAL_PERCENTILE
  csv_path = os.path.join(
      TARGET_DIR,
      f"temperature_responses_co2_p{int(response_percentile*100):02d}_{SCENARIO}.csv",
  )

  df_responsecurves = pd.read_csv(csv_path)

  if co2_emission_adj is None:
    co2_emission_adj = np.zeros(1000)
  if co2_cdr_adj is None:
    co2_cdr_adj = np.zeros(1000)
  if ch4_adj is None:
    ch4_adj = np.zeros(1000)
  if hfc_adj is None:
    hfc_adj = np.zeros(1000)

  print("Running Baseline...")
  f_base = run_fair()

  def get_formatted_df(f_scenario):
    t_base = f_base.temperature.sel(scenario=SCENARIO, layer=0).squeeze(
        drop=True
    )
    t_scen = f_scenario.temperature.sel(scenario=SCENARIO, layer=0).squeeze(
        drop=True
    )
    diff = t_scen - t_base

    percentiles_to_calc = list(set([0.05, 0.50, 0.95, EVAL_PERCENTILE]))
    percentiles_to_calc.sort()

    q = diff.quantile(percentiles_to_calc, dim="config")
    time_dim = [d for d in q.dims if d != "quantile"][0]
    df = q.transpose(time_dim, "quantile").to_pandas()
    df.columns = [f"p{int(col * 100):02d}" for col in df.columns]

    return df.reset_index()

  print("Running Uncompensated Scenario...")
  co2_total_uncomp = co2_emission_adj + co2_cdr_adj
  f_uncomp = run_fair(
      co2_adj=co2_total_uncomp, ch4_adj=ch4_adj, hfc_adj=hfc_adj
  )

  df_uncomp = get_formatted_df(f_uncomp)
  percentile_col = f"p{int(EVAL_PERCENTILE * 100):02d}"

  model_years = df_uncomp.iloc[:, 0].values

  sim_year_cols = [
      str(col) for col in df_responsecurves.columns if str(col).isdigit()
  ]

  print("Loading pre-calculated temperature response shapes...")
  df_responses = {
      year: df_responsecurves[year].values for year in sim_year_cols
  }

  dt = 1
  nonzero_indices = np.nonzero(co2_emission_adj)[0]
  if len(nonzero_indices) == 0:
    print(
        "Warning: co2_emission_adj contains only zeros. No compensation range"
        " defined."
    )
    return {}, df_uncomp, df_uncomp

  start_year = int(nonzero_indices[0]) + START_YEAR
  print(f"\nCompensation Window Determined: {start_year} to {MAX_YEAR}")

  additional_cdr_required = {}
  calculated_cdr_pulse = np.zeros(1000)

  total_cde_emitted = np.sum(co2_emission_adj)
  total_baseline_cdr = np.sum(np.abs(co2_cdr_adj))

  all_target_years = list(range(start_year, MAX_YEAR + 1))

  if not use_solver:

    current_temp_profile = df_uncomp[percentile_col].values.copy()

    for year in all_target_years:
      target_year = year + dt
      if target_year > model_years[-1]:
        target_year = model_years[-1]

      target_idx = np.abs(model_years - target_year).argmin()
      temp_at_target = current_temp_profile[target_idx]

      response_curve = df_responses[str(year)]
      response_factor_at_target = response_curve[target_idx]

      current_additional_cdr_sum = sum(additional_cdr_required.values())
      residual_cdr = total_cde_emitted - (
          total_baseline_cdr + current_additional_cdr_sum
      )

      if year == MAX_YEAR:
        needed_cdr = max(0, residual_cdr)
        current_temp_profile += needed_cdr * response_curve
        additional_cdr_required[year] = needed_cdr
      else:
        if temp_at_target > 0:
          if abs(response_factor_at_target) < 1e-12:
            needed_cdr = 0
          else:
            raw_needed_cdr = -temp_at_target / response_factor_at_target
            needed_cdr = min(raw_needed_cdr, max(0, residual_cdr))

          current_temp_profile += needed_cdr * response_curve
          additional_cdr_required[year] = needed_cdr
        else:
          additional_cdr_required[year] = 0

    # Build final pulse array for the uncompensated fallback
    for year_key, cdr_amount in additional_cdr_required.items():
      idx = int(year_key - START_YEAR)
      if 0 <= idx < 1000:
        calculated_cdr_pulse[idx] = -cdr_amount

  else:

    # Initialize cache variables before the loop
    last_pulse_state = None
    df_eval = None

    for year in all_target_years:
      idx = int(year - START_YEAR)

      current_additional_cdr_sum = np.sum(np.abs(calculated_cdr_pulse))
      residual_cdr = total_cde_emitted - (
          total_baseline_cdr + current_additional_cdr_sum
      )

      if residual_cdr < tolerance:
        if residual_cdr > 1e-9:
          print(
              f"  Budget below tolerance ({residual_cdr:.3f} Gt). Dumping into"
              f" {year} and halting simulation."
          )
          calculated_cdr_pulse[idx] = -residual_cdr
          additional_cdr_required[year] = residual_cdr
        else:
          print(
              "  Mass balance strictly achieved. Halting simulation for"
              " remaining years."
          )
        break

      target_year = min(year + dt, model_years[-1])
      idx_target = np.abs(model_years - target_year).argmin()

      response_curve = df_responses[str(year)]
      response_factor_target = response_curve[idx_target]

      if abs(response_factor_target) < 1e-12:
        additional_cdr_required[year] = 0
        continue

      if last_pulse_state is None or not np.array_equal(
          calculated_cdr_pulse, last_pulse_state
      ):
        print(f"  Evaluating temperature state for {year}...")
        current_co2_total = co2_total_uncomp + calculated_cdr_pulse
        f_eval = run_fair(
            co2_adj=current_co2_total, ch4_adj=ch4_adj, hfc_adj=hfc_adj
        )
        df_eval = get_formatted_df(f_eval)
        last_pulse_state = calculated_cdr_pulse.copy()

      future_temps = df_eval[percentile_col].values[idx_target:]
      if np.max(future_temps) <= 0:
        print(
            "  Future temperature horizon perfectly stabilized (<= 0.0°C)."
            " Halting simulation."
        )
        break

      temp_target = df_eval[percentile_col].values[idx_target]
      if temp_target <= 0:
        additional_cdr_required[year] = 0
        continue

      linear_guess = max(0, -temp_target / response_factor_target)

      if linear_guess < tolerance:
        print(
            f"    Bypassing solver: Linear guess ({linear_guess:.4f}) is below"
            f" tolerance resolution ({tolerance}). Using tolerance instead."
        )

        # Cap the tolerance to ensure we don't exceed the mass balance ceiling
        final_guess = min(tolerance, residual_cdr)

        calculated_cdr_pulse[idx] = -final_guess
        additional_cdr_required[year] = final_guess
        continue

      print(
          f"  Optimizing Non-Linear CDR dose for {year} (Target Temp"
          f" {target_year} = 0.0°C)..."
      )

      low = 0.0
      high = linear_guess * 2.0

      iteration = 0
      best_cdr = high

      while (high - low) > tolerance and iteration < 20:
        iteration += 1
        mid = (low + high) / 2.0

        print(
            f"    Iteration {iteration}/20: Testing CDR = {mid:.3f} (Bounds:"
            f" {low:.3f} to {high:.3f})"
        )

        test_cdr_pulse = calculated_cdr_pulse.copy()
        test_cdr_pulse[idx] = -mid
        test_total_co2 = co2_total_uncomp + test_cdr_pulse

        f_test = run_fair(
            co2_adj=test_total_co2, ch4_adj=ch4_adj, hfc_adj=hfc_adj
        )
        df_test = get_formatted_df(f_test)
        temp_target_test = df_test[percentile_col].values[idx_target]

        if temp_target_test > 0:
          low = mid
        else:
          high = mid
          best_cdr = mid

      # Enforce the tolerance floor on the solver's optimized output
      final_cdr = min(max(best_cdr, tolerance), residual_cdr)

      calculated_cdr_pulse[idx] = -final_cdr
      additional_cdr_required[year] = final_cdr

    current_additional_cdr_sum = sum(additional_cdr_required.values())
    final_residual_cdr = total_cde_emitted - (
        total_baseline_cdr + current_additional_cdr_sum
    )

    if final_residual_cdr > 1e-9:
      # Find the last year where CDR was actively required by the physics solver
      req_years = [y for y, val in additional_cdr_required.items() if val > 0]

      if req_years:
        last_req_year = max(req_years)
        print(
            "  Enforcing Mass Balance: Adding remaining"
            f" {final_residual_cdr:.3f} Gt CDR to the final intervention year"
            f" ({last_req_year})."
        )
        additional_cdr_required[last_req_year] += final_residual_cdr

        idx = int(last_req_year - START_YEAR)
        calculated_cdr_pulse[idx] -= final_residual_cdr
      else:
        # Fallback: No additional CDR was ever required (e.g. massive SLCP
        # overcompensation).
        fallback_year = START_YEAR + OFFSET
        print(
            "  Enforcing Mass Balance: No intervention was required. Adding"
            f" remaining {final_residual_cdr:.3f} Gt CDR to emission year"
            f" ({fallback_year})."
        )
        additional_cdr_required[fallback_year] = (
            additional_cdr_required.get(fallback_year, 0) + final_residual_cdr
        )
        calculated_cdr_pulse[OFFSET] -= final_residual_cdr

  print("Running Final Compensated Scenario...")
  co2_total_comp = co2_total_uncomp + calculated_cdr_pulse
  f_comp = run_fair(co2_adj=co2_total_comp, ch4_adj=ch4_adj, hfc_adj=hfc_adj)
  df_comp = get_formatted_df(f_comp)

  print("Calculation Complete.")
  return additional_cdr_required, df_uncomp, df_comp


def calc_min_ch4(
    co2_emission_adj=None,
    co2_cdr_adj=None,
    ch4_adj=None,
    hfc_adj=None,
    use_solver=True,
    tolerance=TOLERANCE,
):
  """Calculates the minimum CH4 mitigation required to neutralize a CO2 pulse.

  Uses an optional iterative solver to account for the non-linear chemical and
  radiative feedbacks of CH4. Note that to achieve long-term temperature
  stabilization the cumulative mass in co2_emission_adj and co2_cdr_adj must be
  equivalent. Emissions can include CO2, CH4 and HFC. Existing mitigation can
  include CDR, CH4 and HFC.

  Args:
    co2_emission_adj: Array of CO2 emission adjustments.
    co2_cdr_adj: Array of CO2 CDR adjustments.
    ch4_adj: Array of CH4 adjustments.
    hfc_adj: Array of HFC adjustments.
    use_solver: Whether to use the iterative solver.
    tolerance: Convergence tolerance for the solver.

  Returns:
    A tuple of (additional_ch4_required, df_uncomp, df_comp).
  """
  response_percentile = EVAL_PERCENTILE
  csv_path = os.path.join(
      TARGET_DIR,
      f"temperature_responses_ch4_p{int(response_percentile*100):02d}_{SCENARIO}.csv",
  )

  df_responsecurves = pd.read_csv(csv_path)

  if co2_emission_adj is None:
    co2_emission_adj = np.zeros(1000)
  if co2_cdr_adj is None:
    co2_cdr_adj = np.zeros(1000)
  if ch4_adj is None:
    ch4_adj = np.zeros(1000)
  if hfc_adj is None:
    hfc_adj = np.zeros(1000)

  print("Running Baseline...")
  f_base = run_fair()

  def get_formatted_df(f_scenario):
    t_base = f_base.temperature.sel(scenario=SCENARIO, layer=0).squeeze(
        drop=True
    )
    t_scen = f_scenario.temperature.sel(scenario=SCENARIO, layer=0).squeeze(
        drop=True
    )
    diff = t_scen - t_base

    percentiles_to_calc = list(set([0.05, 0.50, 0.95, EVAL_PERCENTILE]))
    percentiles_to_calc.sort()

    q = diff.quantile(percentiles_to_calc, dim="config")
    time_dim = [d for d in q.dims if d != "quantile"][0]
    df = q.transpose(time_dim, "quantile").to_pandas()
    df.columns = [f"p{int(col * 100):02d}" for col in df.columns]

    return df.reset_index()

  print("Running Uncompensated Scenario...")
  co2_total_uncomp = co2_emission_adj + co2_cdr_adj
  f_uncomp = run_fair(
      co2_adj=co2_total_uncomp, ch4_adj=ch4_adj, hfc_adj=hfc_adj
  )

  df_uncomp = get_formatted_df(f_uncomp)
  percentile_col = f"p{int(EVAL_PERCENTILE * 100):02d}"

  model_years = df_uncomp.iloc[:, 0].values
  sim_year_cols = [
      str(col) for col in df_responsecurves.columns if str(col).isdigit()
  ]
  df_responses = {
      year: df_responsecurves[year].values for year in sim_year_cols
  }

  nonzero_indices = np.nonzero(co2_emission_adj)[0]
  if len(nonzero_indices) == 0:
    print("Warning: co2_emission_adj contains only zeros.")
    return {}, df_uncomp, df_uncomp

  start_year = int(nonzero_indices[0]) + START_YEAR
  print(f"\nCompensation Window Determined: {start_year} to {MAX_YEAR}")

  additional_ch4_required = {}
  calculated_ch4_pulse = np.zeros(1000)

  total_co2_emissions = np.sum(co2_emission_adj)
  cum_co2_cdr = np.cumsum(np.abs(co2_cdr_adj))

  all_target_years = list(range(start_year, MAX_YEAR + 1))
  dt = 1

  if not use_solver:

    current_temp_profile = df_uncomp[percentile_col].values.copy()

    for year in all_target_years:
      idx = int(year - START_YEAR)

      if 0 <= idx < len(cum_co2_cdr) and cum_co2_cdr[idx] >= (
          total_co2_emissions - 1e-9
      ):
        print(
            "  Full CO2 compensation achieved via CDR. Halting CH4 mitigation."
        )
        break

      target_year = min(year + dt, model_years[-1])
      target_idx = np.abs(model_years - target_year).argmin()
      temp_at_target = current_temp_profile[target_idx]

      response_curve = df_responses[str(year)]
      response_factor_at_target = response_curve[target_idx]

      if temp_at_target > 0:
        if abs(response_factor_at_target) < 1e-12:
          needed_ch4 = 0
        else:
          needed_ch4 = max(0, -temp_at_target / response_factor_at_target)

        current_temp_profile += needed_ch4 * response_curve
        additional_ch4_required[year] = needed_ch4
      else:
        additional_ch4_required[year] = 0

    # Build final pulse array
    for year_key, ch4_amount in additional_ch4_required.items():
      idx = int(year_key - START_YEAR)
      if 0 <= idx < 1000:
        calculated_ch4_pulse[idx] = -ch4_amount

  else:

    # Initialize cache variables before the loop
    last_pulse_state = None
    df_eval = None

    for year in all_target_years:
      idx = int(year - START_YEAR)

      # OPTIMIZATION 1: The Bridging Cutoff (Break Loop)
      if 0 <= idx < len(cum_co2_cdr) and cum_co2_cdr[idx] >= (
          total_co2_emissions - 1e-9
      ):
        print(
            "  Full CO2 compensation achieved via CDR. Halting CH4 mitigation."
        )
        break

      target_year = min(year + dt, model_years[-1])
      idx_target = np.abs(model_years - target_year).argmin()

      # OPTIMIZATION 2: The Zero-Efficacy Bypass (Pre-Model Check)
      response_curve = df_responses[str(year)]
      response_factor_target = response_curve[idx_target]

      if abs(response_factor_target) < 1e-12:
        additional_ch4_required[year] = 0
        continue

      # THE SILENT EVALUATION
      if last_pulse_state is None or not np.array_equal(
          calculated_ch4_pulse, last_pulse_state
      ):
        print(f"  Evaluating temperature state for {year}...")
        current_ch4_total = ch4_adj + calculated_ch4_pulse
        f_eval = run_fair(
            co2_adj=co2_total_uncomp, ch4_adj=current_ch4_total, hfc_adj=hfc_adj
        )
        df_eval = get_formatted_df(f_eval)
        last_pulse_state = calculated_ch4_pulse.copy()

      # OPTIMIZATION 3: The Global Future Horizon (Break Loop)
      future_temps = df_eval[percentile_col].values[idx_target:]
      if np.max(future_temps) <= 0:
        print(
            "  Future temperature horizon perfectly stabilized (<= 0.0°C)."
            " Halting simulation."
        )
        break

      # OPTIMIZATION 4: The Local Temperature Floor (Continue)
      temp_target = df_eval[percentile_col].values[idx_target]
      if temp_target <= 0:
        additional_ch4_required[year] = 0
        continue

      # OPTIMIZATION 5: The Resolution Bypass (Continue)
      linear_guess = max(0, -temp_target / response_factor_target)
      if linear_guess < tolerance:
        print(
            f"    Bypassing solver: Linear guess ({linear_guess:.4f}) is below"
            f" tolerance resolution ({tolerance}). Using tolerance instead."
        )
        calculated_ch4_pulse[idx] = -tolerance
        additional_ch4_required[year] = tolerance
        continue

      print(
          f"  Optimizing Non-Linear CH4 dose for {year} (Target Temp"
          f" {target_year} = 0.0°C)..."
      )

      low = 0.0
      high = linear_guess * 2.0

      iteration = 0
      best_ch4 = high

      while (high - low) > tolerance and iteration < 20:
        iteration += 1
        mid = (low + high) / 2.0

        print(
            f"    Iteration {iteration}/20: Testing CH4 = {mid:.3f} (Bounds:"
            f" {low:.3f} to {high:.3f})"
        )

        test_ch4_pulse = calculated_ch4_pulse.copy()
        test_ch4_pulse[idx] = -mid
        test_total_ch4 = ch4_adj + test_ch4_pulse

        f_test = run_fair(
            co2_adj=co2_total_uncomp, ch4_adj=test_total_ch4, hfc_adj=hfc_adj
        )
        df_test = get_formatted_df(f_test)
        temp_target_test = df_test[percentile_col].values[idx_target]

        if temp_target_test > 0:
          low = mid
        else:
          high = mid
          best_ch4 = mid

      # Enforce the tolerance floor on the solver's optimized output
      final_ch4 = max(best_ch4, tolerance)

      calculated_ch4_pulse[idx] = -final_ch4
      additional_ch4_required[year] = final_ch4

  print("Running Final Compensated Scenario...")
  ch4_total_comp = ch4_adj + calculated_ch4_pulse
  f_comp = run_fair(
      co2_adj=co2_total_uncomp, ch4_adj=ch4_total_comp, hfc_adj=hfc_adj
  )
  df_comp = get_formatted_df(f_comp)

  print("Calculation Complete.")
  return additional_ch4_required, df_uncomp, df_comp


def calc_min_hfc(
    co2_emission_adj=None,
    co2_cdr_adj=None,
    ch4_adj=None,
    hfc_adj=None,
    hfc_specie=HFC_SPECIE,
    use_solver=True,
    tolerance=TOLERANCE,
):
  """Calculates the minimum HFC mitigation required to neutralize a CO2 pulse.

  Uses an optional iterative solver to account for non-linear chemical and
  radiative feedbacks.

  Args:
    co2_emission_adj: Array of CO2 emission adjustments.
    co2_cdr_adj: Array of CO2 CDR adjustments.
    ch4_adj: Array of CH4 adjustments.
    hfc_adj: Array of HFC adjustments.
    hfc_specie: Name of HFC species to adjust.
    use_solver: Whether to use the iterative solver.
    tolerance: Convergence tolerance for the solver.

  Returns:
    A tuple of (additional_hfc_required, df_uncomp, df_comp).
  """
  response_percentile = EVAL_PERCENTILE
  csv_path = os.path.join(
      TARGET_DIR,
      f"temperature_responses_{hfc_specie}_p{int(response_percentile*100):02d}_{SCENARIO}.csv",
  )

  df_responsecurves = pd.read_csv(csv_path)

  if co2_emission_adj is None:
    co2_emission_adj = np.zeros(1000)
  if co2_cdr_adj is None:
    co2_cdr_adj = np.zeros(1000)
  if ch4_adj is None:
    ch4_adj = np.zeros(1000)
  if hfc_adj is None:
    hfc_adj = np.zeros(1000)

  print("Running Baseline...")
  f_base = run_fair(hfc_specie=hfc_specie)

  def get_formatted_df(f_scenario):
    t_base = f_base.temperature.sel(scenario=SCENARIO, layer=0).squeeze(
        drop=True
    )
    t_scen = f_scenario.temperature.sel(scenario=SCENARIO, layer=0).squeeze(
        drop=True
    )
    diff = t_scen - t_base

    percentiles_to_calc = list(set([0.05, 0.50, 0.95, EVAL_PERCENTILE]))
    percentiles_to_calc.sort()

    q = diff.quantile(percentiles_to_calc, dim="config")
    time_dim = [d for d in q.dims if d != "quantile"][0]
    df = q.transpose(time_dim, "quantile").to_pandas()
    df.columns = [f"p{int(col * 100):02d}" for col in df.columns]

    return df.reset_index()

  print("Running Uncompensated Scenario...")
  co2_total_uncomp = co2_emission_adj + co2_cdr_adj
  f_uncomp = run_fair(
      co2_adj=co2_total_uncomp,
      ch4_adj=ch4_adj,
      hfc_adj=hfc_adj,
      hfc_specie=hfc_specie,
  )

  df_uncomp = get_formatted_df(f_uncomp)
  percentile_col = f"p{int(EVAL_PERCENTILE * 100):02d}"

  model_years = df_uncomp.iloc[:, 0].values
  sim_year_cols = [
      str(col) for col in df_responsecurves.columns if str(col).isdigit()
  ]
  df_responses = {
      year: df_responsecurves[year].values for year in sim_year_cols
  }

  nonzero_indices = np.nonzero(co2_emission_adj)[0]
  if len(nonzero_indices) == 0:
    print("Warning: co2_emission_adj contains only zeros.")
    return {}, df_uncomp, df_uncomp

  start_year = int(nonzero_indices[0]) + START_YEAR
  print(f"\nCompensation Window Determined: {start_year} to {MAX_YEAR}")

  additional_hfc_required = {}
  calculated_hfc_pulse = np.zeros(1000)

  total_co2_emissions = np.sum(co2_emission_adj)
  cum_co2_cdr = np.cumsum(np.abs(co2_cdr_adj))

  all_target_years = list(range(start_year, MAX_YEAR + 1))
  dt = 1

  if not use_solver:
    # FAST LINEAR APPROXIMATION ROUTINE
    current_temp_profile = df_uncomp[percentile_col].values.copy()

    for year in all_target_years:
      idx = int(year - START_YEAR)

      if 0 <= idx < len(cum_co2_cdr) and cum_co2_cdr[idx] >= (
          total_co2_emissions - 1e-9
      ):
        print(
            "  Full CO2 compensation achieved via CDR. Halting HFC mitigation."
        )
        break

      target_year = min(year + dt, model_years[-1])
      target_idx = np.abs(model_years - target_year).argmin()
      temp_at_target = current_temp_profile[target_idx]

      response_curve = df_responses[str(year)]
      response_factor_at_target = response_curve[target_idx]

      if temp_at_target > 0:
        if abs(response_factor_at_target) < 1e-12:
          needed_hfc = 0
        else:
          needed_hfc = max(0, -temp_at_target / response_factor_at_target)

        current_temp_profile += needed_hfc * response_curve
        additional_hfc_required[year] = needed_hfc
      else:
        additional_hfc_required[year] = 0

    for year_key, hfc_amount in additional_hfc_required.items():
      idx = int(year_key - START_YEAR)
      if 0 <= idx < 1000:
        calculated_hfc_pulse[idx] = -hfc_amount

  else:
    # NON-LINEAR BINARY SEARCH SOLVER ROUTINE
    last_pulse_state = None
    df_eval = None

    for year in all_target_years:
      idx = int(year - START_YEAR)

      if 0 <= idx < len(cum_co2_cdr) and cum_co2_cdr[idx] >= (
          total_co2_emissions - 1e-9
      ):
        print(
            "  Full CO2 compensation achieved via CDR. Halting HFC mitigation."
        )
        break

      target_year = min(year + dt, model_years[-1])
      idx_target = np.abs(model_years - target_year).argmin()

      response_curve = df_responses[str(year)]
      response_factor_target = response_curve[idx_target]

      if abs(response_factor_target) < 1e-12:
        additional_hfc_required[year] = 0
        continue

      if last_pulse_state is None or not np.array_equal(
          calculated_hfc_pulse, last_pulse_state
      ):
        print(f"  Evaluating temperature state for {year}...")
        current_hfc_total = hfc_adj + calculated_hfc_pulse
        f_eval = run_fair(
            co2_adj=co2_total_uncomp,
            ch4_adj=ch4_adj,
            hfc_adj=current_hfc_total,
            hfc_specie=hfc_specie,
        )
        df_eval = get_formatted_df(f_eval)
        last_pulse_state = calculated_hfc_pulse.copy()

      future_temps = df_eval[percentile_col].values[idx_target:]
      if np.max(future_temps) <= 0:
        print(
            "  Future temperature horizon perfectly stabilized (<= 0.0°C)."
            " Halting simulation."
        )
        break

      temp_target = df_eval[percentile_col].values[idx_target]
      if temp_target <= 0:
        additional_hfc_required[year] = 0
        continue

      linear_guess = max(0, -temp_target / response_factor_target)
      if linear_guess < tolerance:
        print(
            f"    Bypassing solver: Linear guess ({linear_guess:.4f}) is below"
            f" tolerance resolution ({tolerance}). Using tolerance instead."
        )
        calculated_hfc_pulse[idx] = -tolerance
        additional_hfc_required[year] = tolerance
        continue

      print(
          f"  Optimizing Non-Linear HFC dose for {year} (Target Temp"
          f" {target_year} = 0.0°C)..."
      )

      low = 0.0
      high = linear_guess * 2.0

      iteration = 0
      best_hfc = high

      while (high - low) > tolerance and iteration < 20:
        iteration += 1
        mid = (low + high) / 2.0

        print(
            f"    Iteration {iteration}/20: Testing HFC = {mid:.3f} (Bounds:"
            f" {low:.3f} to {high:.3f})"
        )

        test_hfc_pulse = calculated_hfc_pulse.copy()
        test_hfc_pulse[idx] = -mid
        test_total_hfc = hfc_adj + test_hfc_pulse

        f_test = run_fair(
            co2_adj=co2_total_uncomp,
            ch4_adj=ch4_adj,
            hfc_adj=test_total_hfc,
            hfc_specie=hfc_specie,
        )
        df_test = get_formatted_df(f_test)
        temp_target_test = df_test[percentile_col].values[idx_target]

        if temp_target_test > 0:
          low = mid
        else:
          high = mid
          best_hfc = mid

      final_hfc = max(best_hfc, tolerance)
      calculated_hfc_pulse[idx] = -final_hfc
      additional_hfc_required[year] = final_hfc

  print("Running Final Compensated Scenario...")
  hfc_total_comp = hfc_adj + calculated_hfc_pulse
  f_comp = run_fair(
      co2_adj=co2_total_uncomp,
      ch4_adj=ch4_adj,
      hfc_adj=hfc_total_comp,
      hfc_specie=hfc_specie,
  )
  df_comp = get_formatted_df(f_comp)

  print("Calculation Complete.")
  return additional_hfc_required, df_uncomp, df_comp
