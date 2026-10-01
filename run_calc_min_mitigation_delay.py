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

"""Calculates minimum mitigation delay curves and tables."""

import os
import fair_utils
import pandas as pd


def execute_calc_min_mitigation_delay(target_cdr_years=None):
  """Executes binary search for minimum mitigation for Table 2 and S1.

  Args:
    target_cdr_years: List of target CDR years to evaluate.

  Returns:
    DataFrame containing the minimum mitigation calculation results.
  """
  if target_cdr_years is None:
    target_cdr_years = list(
        range(fair_utils.TARGET_YEAR + 1, fair_utils.TARGET_YEAR + 21)
    )

  print("Running Baseline...")
  f_base = fair_utils.run_fair()
  t_base = f_base.temperature.sel(
      scenario=fair_utils.SCENARIO, layer=0
  ).squeeze(drop=True)

  results_data = []

  print(
      f"\nStarting binary search (Percentile: {fair_utils.EVAL_PERCENTILE})..."
  )
  print("*" * 80)

  for target_cdr_year in target_cdr_years:
    delay_years = target_cdr_year - fair_utils.TARGET_YEAR

    req_ch4, iter_ch4 = fair_utils.calc_min_mitigation_delay(
        delay_years, t_base, target_gas="CH4", search_low=0.0, search_high=30.0
    )

    req_hfc, iter_hfc = fair_utils.calc_min_mitigation_delay(
        delay_years,
        t_base,
        target_gas="HFC",
        search_low=150.0,
        search_high=450.0,
    )

    results_data.append({
        "Emission_Year": fair_utils.TARGET_YEAR,
        "Delayed_CDR_Year": target_cdr_year,
        "Delay_Period_Years": delay_years,
        "Required_CH4_Mt": req_ch4,
        "Required_HFC_Mt": req_hfc,
        "Iterations_CH4": iter_ch4,
        "Iterations_HFC": iter_hfc,
    })

    print(f"CDR {target_cdr_year} ({delay_years}-yr delay):")
    print(f"  > CH₄ Mitigation: {req_ch4:.3f} Mt (Took {iter_ch4} iterations)")
    print(f"  > HFC Mitigation: {req_hfc:.4f} Mt (Took {iter_hfc} iterations)")

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
      f"min_mitigation_p{int(fair_utils.EVAL_PERCENTILE*100)}_{year_suffix}_"
      f"{fair_utils.SCENARIO}.csv",
  )
  df_results.to_csv(csv_path, index=False)
  print(f"\nSuccess! Results saved to: {csv_path}")

  return df_results


if __name__ == "__main__":
  execute_calc_min_mitigation_delay()
