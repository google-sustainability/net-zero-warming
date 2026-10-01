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

"""Calculates minimum mitigation delay curves based on radiative forcing."""

import os
import fair_utils
import numpy as np
import pandas as pd


def calc_min_mitigation_delay_forcing(
    delay_years,
    forcing_base,
    target_gas="CH4",
    search_low=0.0,
    search_high=50.0,
):
  """Calculates the minimum mitigation required based on Radiative Forcing.

  Args:
    delay_years: Number of years delay between emission and CDR.
    forcing_base: Baseline forcing data array.
    target_gas: Target gas to mitigate ('CH4' or 'HFC').
    search_low: Lower bound for binary search.
    search_high: Upper bound for binary search.

  Returns:
    Tuple of (required_mitigation, iterations_count).
  """
  idx_start_eval = (fair_utils.TARGET_YEAR + 1) - fair_utils.MIN_YEAR
  idx_end_eval = (fair_utils.TARGET_YEAR + delay_years) - fair_utils.MIN_YEAR

  low, high = search_low, search_high
  iteration = 0

  while (high - low) > fair_utils.TOLERANCE:
    iteration += 1
    mid_val = (low + high) / 2.0

    co2_pulse = np.zeros(1000)
    co2_pulse[fair_utils.OFFSET] = 1.0
    co2_pulse[fair_utils.OFFSET + delay_years] = -1.0

    mitigation_pulse = np.zeros(1000)
    mitigation_pulse[fair_utils.OFFSET] = -mid_val

    if target_gas.upper() == "CH4":
      f_scenario = fair_utils.run_fair(
          co2_adj=co2_pulse, ch4_adj=mitigation_pulse
      )
    elif target_gas.upper() == "HFC":
      f_scenario = fair_utils.run_fair(
          co2_adj=co2_pulse, hfc_adj=mitigation_pulse
      )
    else:
      raise ValueError("target_gas must be either 'CH4' or 'HFC'")

    # Extract forcing sum instead of temperature
    forcing_scenario = f_scenario.forcing_sum.sel(
        scenario=fair_utils.SCENARIO
    ).squeeze(drop=True)

    # Calculate the forcing difference and evaluate at the target percentile
    forcing_diff = (forcing_scenario - forcing_base).quantile(
        fair_utils.EVAL_PERCENTILE, dim="config"
    )

    # Check the maximum forcing anomaly in the target evaluation window
    max_forcing_anomaly = (
        forcing_diff[idx_start_eval : idx_end_eval + 1].max().values
    )

    if max_forcing_anomaly > 0:
      low = mid_val
    else:
      high = mid_val

  return high, iteration


def execute_calc_min_mitigation_delay_forcing(target_cdr_years=None):
  """Executes the binary search for minimum mitigation based on forcing.

  Args:
    target_cdr_years: List of target CDR years to evaluate.

  Returns:
    DataFrame containing the minimum mitigation calculation results.
  """
  if target_cdr_years is None:
    target_cdr_years = list(
        range(fair_utils.TARGET_YEAR + 1, fair_utils.TARGET_YEAR + 21)
    )

  print("Running Baseline for Forcing...")
  f_base = fair_utils.run_fair()

  # Extract baseline forcing sum
  forcing_base = f_base.forcing_sum.sel(
      scenario=fair_utils.SCENARIO
  ).squeeze(drop=True)

  results_data = []

  print(
      "\nStarting forcing-based binary search (Percentile:"
      f" {fair_utils.EVAL_PERCENTILE})..."
  )
  print("*" * 80)

  for target_cdr_year in target_cdr_years:
    delay_years = target_cdr_year - fair_utils.TARGET_YEAR

    req_ch4, iter_ch4 = calc_min_mitigation_delay_forcing(
        delay_years,
        forcing_base,
        target_gas="CH4",
        search_low=0.0,
        search_high=50.0,
    )

    results_data.append({
        "Emission_Year": fair_utils.TARGET_YEAR,
        "Delayed_CDR_Year": target_cdr_year,
        "Delay_Period_Years": delay_years,
        "Required_CH4_Mt_Forcing": req_ch4,
        "Iterations_CH4": iter_ch4,
    })

    print(f"CDR {target_cdr_year} ({delay_years}-yr delay):")
    print(
        f"  > CH4 Mitigation (Forcing): {req_ch4:.3f} Mt (Took {iter_ch4}"
        " iterations)"
    )

  print("\n" + "*" * 80)
  df_results = pd.DataFrame(results_data)
  print(df_results.to_string(index=False))

  os.makedirs(fair_utils.TARGET_DIR, exist_ok=True)
  min_cdr_year = min(target_cdr_years)
  max_cdr_year = max(target_cdr_years)
  year_suffix = (
      f"{min_cdr_year}"
      if min_cdr_year == max_cdr_year
      else f"{min_cdr_year}_{max_cdr_year}"
  )

  csv_path = os.path.join(
      fair_utils.TARGET_DIR,
      f"min_mitigation_forcing_p{int(fair_utils.EVAL_PERCENTILE*100)}_{year_suffix}_"
      f"{fair_utils.SCENARIO}.csv",
  )
  df_results.to_csv(csv_path, index=False)
  print(f"\nSuccess! Forcing results saved to: {csv_path}")

  return df_results


if __name__ == "__main__":
  execute_calc_min_mitigation_delay_forcing()
