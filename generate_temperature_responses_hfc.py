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

"""Generates temperature response curves for HFC pulses."""

import os
import fair_utils
import numpy as np
import pandas as pd


def execute_temperature_responses_hfc(
    sim_years=None, unit_mag=1.0, hfc_specie=fair_utils.HFC_SPECIE
):
  """Simulates and exports temperature response curves for anchor years.

  Args:
    sim_years: List of simulation years. Defaults to START_YEAR through 2500.
    unit_mag: Unit magnitude of the pulse.
    hfc_specie: HFC species name to simulate.

  Returns:
    DataFrame containing the simulated response curves.
  """
  if sim_years is None:
    sim_years = list(range(fair_utils.START_YEAR, 2501))

  simulated_curves = {}

  response_percentile = fair_utils.EVAL_PERCENTILE

  print("Running Baseline...")
  # Pass the hfc_specie to the baseline to ensure consistency
  f_base = fair_utils.run_fair(hfc_specie=hfc_specie)
  t_base = f_base.temperature.sel(
      scenario=fair_utils.SCENARIO, layer=0
  ).squeeze(drop=True)

  for year in sim_years:
    print(f"  Simulating {hfc_specie} removal pulse in year {year}...")

    offset = year - fair_utils.START_YEAR
    unit_pulse = np.zeros(1000)
    unit_pulse[offset] = -unit_mag

    # Pass the pulse as an hfc_adj along with the target species
    f_unit = fair_utils.run_fair(hfc_adj=unit_pulse, hfc_specie=hfc_specie)
    t_pulse_run = f_unit.temperature.sel(
        scenario=fair_utils.SCENARIO, layer=0
    ).squeeze(drop=True)
    t_diff = (
        (t_pulse_run - t_base)
        .quantile(response_percentile, dim="config")
        .values
    )

    simulated_curves[year] = t_diff / unit_mag

  # Include the specific HFC species in the filename
  filename = (
      f"temperature_responses_{hfc_specie}_p{int(response_percentile*100):02d}_"
      f"{fair_utils.SCENARIO}.csv"
  )
  csv_path = os.path.join(fair_utils.TARGET_DIR, filename)

  print(f"\nSaving simulated curves to: {csv_path}")
  df_save = pd.DataFrame(simulated_curves)

  os.makedirs(os.path.dirname(csv_path), exist_ok=True)
  df_save.to_csv(csv_path, index=False)
  print("Save complete!")

  return df_save


if __name__ == "__main__":
  execute_temperature_responses_hfc()
