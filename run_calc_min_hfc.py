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

"""Calculates minimum required HFC mitigation and plots profiles."""

import os
import fair_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def execute_calc_min_hfc(
    co2_emission_adj=None,
    co2_cdr_adj=None,
    ch4_adj=None,
    hfc_adj=None,
    hfc_specie=fair_utils.HFC_SPECIE,
    gwp_horizon=100,
    use_solver=True,
    tolerance=fair_utils.TOLERANCE,
    skip_calc=False,
):
  """Executes the HFC mitigation calculation.

  Args:
    co2_emission_adj: Array of CO2 emissions adjustments.
    co2_cdr_adj: Array of CO2 CDR adjustments.
    ch4_adj: Array of CH4 adjustments.
    hfc_adj: Array of HFC adjustments.
    hfc_specie: Name of the HFC species.
    gwp_horizon: GWP time horizon (20 or 100).
    use_solver: Whether to use iterative solver.
    tolerance: Convergence tolerance.
    skip_calc: Set True to bypass simulation and plot from saved CSV outputs.

  Returns:
    Tuple of (additional_hfc_req, df_uncomp, df_comp).
  """
  if co2_emission_adj is None:
    co2_emission_adj = np.zeros(1000)
    co2_emission_adj[fair_utils.OFFSET] = 1.0

  if co2_cdr_adj is None:
    co2_cdr_adj = np.zeros(1000)
    co2_cdr_adj[fair_utils.OFFSET + 10] = -1.0

  if ch4_adj is None:
    ch4_adj = np.zeros(1000)

  if hfc_adj is None:
    hfc_adj = np.zeros(1000)

  gwp_hfc = (
      fair_utils.GWP100_HFC if gwp_horizon == 100 else fair_utils.GWP20_HFC
  )

  uncomp_path = os.path.join(
      fair_utils.TARGET_DIR, 'temperature_uncompensated_min_hfc.csv'
  )
  comp_path = os.path.join(
      fair_utils.TARGET_DIR, 'temperature_compensated_min_hfc.csv'
  )
  schedule_path = os.path.join(fair_utils.TARGET_DIR, 'min_hfc.csv')

  if skip_calc:
    print('Skipping calculation. Loading previously generated outputs...')
    try:
      df_uncomp = pd.read_csv(uncomp_path)
      df_comp = pd.read_csv(comp_path)
      print('Loaded temperature profiles successfully.')
    except FileNotFoundError as exc:
      raise FileNotFoundError(
          'Could not find previous CSV outputs. Please run with skip_calc=False'
          ' first.'
      ) from exc

    # Load the HFC mitigation schedule if it exists
    if os.path.exists(schedule_path):
      df_hfc_schedule = pd.read_csv(schedule_path)
      additional_hfc_req = dict(
          zip(df_hfc_schedule['Year'], df_hfc_schedule['Required_HFC_Mt'])
      )
      print(f'Loaded {hfc_specie} mitigation schedule successfully.')
    else:
      additional_hfc_req = {}

  else:
    print(
        f'Calculating Minimum Required {hfc_specie} Mitigation (Solver Enabled:'
        f' {use_solver})...'
    )
    additional_hfc_req, df_uncomp, df_comp = fair_utils.calc_min_hfc(
        co2_emission_adj=co2_emission_adj,
        co2_cdr_adj=co2_cdr_adj,
        ch4_adj=ch4_adj,
        hfc_adj=hfc_adj,
        hfc_specie=hfc_specie,
        use_solver=use_solver,
        tolerance=tolerance,
    )

    df_uncomp.to_csv(uncomp_path, index=False)
    print(f'Uncompensated profile saved to: {uncomp_path}')

    df_comp.to_csv(comp_path, index=False)
    print(f'Compensated profile saved to: {comp_path}')

    if additional_hfc_req:
      df_hfc_schedule = pd.DataFrame(
          list(additional_hfc_req.items()), columns=['Year', 'Required_HFC_Mt']
      )
      df_hfc_schedule.to_csv(schedule_path, index=False)
      print(f'{hfc_specie} Mitigation Schedule saved to: {schedule_path}')
    else:
      # Clean up old file if the new run requires no HFC mitigation
      if os.path.exists(schedule_path):
        os.remove(schedule_path)

  # Plotting Logic
  time_col = df_uncomp.columns[0]

  for df in [df_uncomp, df_comp]:
    for col in df.columns:
      if col != time_col:
        df[col] *= 1000

  col_uncomp = '#E15759'
  col_comp = '#0072B2'
  col_hfc = '#9467BD'  # Purple for HFC to distinguish from CH4
  col_cdr = '#009E73'

  _, (ax1, ax2) = plt.subplots(
      2, 1, figsize=(8, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]}
  )

  def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

  style_axes(ax1)
  ax1.axhline(0, color='gray', linestyle='-', linewidth=1.2, zorder=1)

  ax1.plot(
      df_uncomp[time_col],
      df_uncomp['p50'],
      color=col_uncomp,
      label='Emissions + Durable CDR',
      linewidth=2.5,
      zorder=3,
  )
  ax1.fill_between(
      df_uncomp[time_col],
      df_uncomp['p05'],
      df_uncomp['p95'],
      color=col_uncomp,
      alpha=0.15,
      zorder=2,
  )

  ax1.plot(
      df_comp[time_col],
      df_comp['p50'],
      color=col_comp,
      label=f'Emissions + Durable CDR\n+ {hfc_specie} Mitigation',
      linewidth=2.5,
      zorder=3,
  )
  ax1.fill_between(
      df_comp[time_col],
      df_comp['p05'],
      df_comp['p95'],
      color=col_comp,
      alpha=0.15,
      zorder=2,
  )

  ax1.set_ylabel('Δ Temperature (10⁻³ °C)', fontsize=12, weight='medium')
  ax1.tick_params(axis='both', which='major', labelsize=12)
  ax1.legend(frameon=False, fontsize=12, loc='upper right')

  style_axes(ax2)

  years_full = np.arange(
      fair_utils.START_YEAR, fair_utils.START_YEAR + 1000
  )

  # 1. Cumulative CDR Line
  cdr_array = np.zeros(1000)
  cdr_indices = np.where(co2_cdr_adj < 0)[0]
  if len(cdr_indices) > 0:
    cdr_array[cdr_indices] = -co2_cdr_adj[cdr_indices]

  cdr_cum = np.cumsum(cdr_array)

  if np.any(cdr_cum > 0):
    ax2.fill_between(
        years_full, 0, cdr_cum, color=col_cdr, alpha=0.15, step='post', zorder=3
    )
    ax2.plot(
        years_full,
        cdr_cum,
        color=col_cdr,
        drawstyle='steps-post',
        linewidth=2.5,
        label='Durable CDR',
        zorder=4,
    )

  # 2. Cumulative HFC Mitigation Line
  hfc_array = np.zeros(1000)
  if additional_hfc_req:
    for year, val in additional_hfc_req.items():
      idx = int(year - fair_utils.START_YEAR)
      if 0 <= idx < 1000:
        hfc_array[idx] += val * gwp_hfc / 1000000

  hfc_cum = np.cumsum(hfc_array)

  if np.any(hfc_cum > 0):
    ax2.fill_between(
        years_full, 0, hfc_cum, color=col_hfc, alpha=0.15, step='post', zorder=3
    )
    ax2.plot(
        years_full,
        hfc_cum,
        color=col_hfc,
        drawstyle='steps-post',
        linewidth=2.5,
        label=f'{hfc_specie} Mitigation',
        zorder=4,
    )

  ax2.set_ylabel(
      f'Cumulative Mitigation\n(Gt CO₂e, GWP{gwp_horizon})',
      fontsize=12,
      weight='medium',
  )
  ax2.set_xlabel('Year', fontsize=12, weight='medium')
  ax2.tick_params(axis='both', which='major', labelsize=12)

  ax2.set_xlim(2025, 2100)

  max_y = max([np.max(cdr_cum), np.max(hfc_cum), 0.05]) * 1.15
  ax2.set_ylim(0, max_y)

  ax2.legend(frameon=False, fontsize=12, loc='lower right', ncol=1)

  plt.tight_layout()
  plt.subplots_adjust(hspace=0.08)

  final_save_path = os.path.join(
      fair_utils.TARGET_DIR, f'min_hfc_solver{use_solver}.png'
  )
  plt.savefig(final_save_path, dpi=300, bbox_inches='tight', facecolor='white')
  print(f'Figure saved to: {final_save_path}')

  plt.show()


if __name__ == '__main__':
  execute_calc_min_hfc(skip_calc=False)
