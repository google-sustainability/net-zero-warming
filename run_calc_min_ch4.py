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

"""Calculates minimum required CH4 mitigation and generates Figure 2."""

import os
import fair_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def execute_calc_min_ch4(
    co2_emission_adj=None,
    co2_cdr_adj=None,
    ch4_adj=None,
    hfc_adj=None,
    gwp_horizon=100,
    use_solver=True,
    tolerance=fair_utils.TOLERANCE,
    skip_calc=False,
):
  """Executes the CH4 mitigation calculation and generates Figure 2.

  Args:
    co2_emission_adj: Array of CO2 emissions adjustments.
    co2_cdr_adj: Array of CO2 CDR adjustments.
    ch4_adj: Array of CH4 adjustments.
    hfc_adj: Array of HFC adjustments.
    gwp_horizon: GWP time horizon (20 or 100).
    use_solver: Whether to use iterative solver.
    tolerance: Convergence tolerance.
    skip_calc: Set True to bypass simulation and plot from saved CSV outputs.

  Returns:
    Tuple of (additional_ch4_req, df_uncomp, df_comp).
  """
  # Set default dummy arrays if none are provided
  if co2_emission_adj is None:
    co2_emission_adj = np.zeros(1000)
    co2_emission_adj[fair_utils.OFFSET] = 1.0  # 1 Gt pulse at target year

  if co2_cdr_adj is None:
    co2_cdr_adj = np.zeros(1000)
    # Example: 1 Gt of CDR achieved exactly 20 years later
    co2_cdr_adj[fair_utils.OFFSET + 10] = -1.0

  if ch4_adj is None:
    ch4_adj = np.zeros(1000)

  if hfc_adj is None:
    hfc_adj = np.zeros(1000)

  gwp_ch4 = (
      fair_utils.GWP100_CH4 if gwp_horizon == 100 else fair_utils.GWP20_CH4
  )

  uncomp_path = os.path.join(
      fair_utils.TARGET_DIR, 'temperature_uncompensated_min_ch4.csv'
  )
  comp_path = os.path.join(
      fair_utils.TARGET_DIR, 'temperature_compensated_min_ch4.csv'
  )
  schedule_path = os.path.join(fair_utils.TARGET_DIR, 'min_ch4.csv')

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

    # Load the CH4 mitigation schedule if it exists
    if os.path.exists(schedule_path):
      df_ch4_schedule = pd.read_csv(schedule_path)
      additional_ch4_req = dict(
          zip(df_ch4_schedule['Year'], df_ch4_schedule['Required_CH4_Mt'])
      )
      print('Loaded CH4 mitigation schedule successfully.')
    else:
      additional_ch4_req = {}

  else:
    print(
        'Calculating Minimum Required CH4 Mitigation (Solver Enabled:'
        f' {use_solver})...'
    )
    additional_ch4_req, df_uncomp, df_comp = fair_utils.calc_min_ch4(
        co2_emission_adj=co2_emission_adj,
        co2_cdr_adj=co2_cdr_adj,
        ch4_adj=ch4_adj,
        hfc_adj=hfc_adj,
        use_solver=use_solver,
        tolerance=tolerance,
    )

    df_uncomp.to_csv(uncomp_path, index=False)
    print(f'Uncompensated profile saved to: {uncomp_path}')

    df_comp.to_csv(comp_path, index=False)
    print(f'Compensated profile saved to: {comp_path}')

    if additional_ch4_req:
      df_ch4_schedule = pd.DataFrame(
          list(additional_ch4_req.items()), columns=['Year', 'Required_CH4_Mt']
      )
      df_ch4_schedule.to_csv(schedule_path, index=False)
      print(f'CH4 Mitigation Schedule saved to: {schedule_path}')
    else:
      # Clean up old file if the new run requires no CH4 mitigation
      if os.path.exists(schedule_path):
        os.remove(schedule_path)

  # Plotting Logic
  time_col = df_uncomp.columns[0]

  # Scale the temperature values to 10^-3 °C for plotting
  for df in [df_uncomp, df_comp]:
    for col in df.columns:
      if col != time_col:
        df[col] *= 1000

  col_uncomp = '#E15759'
  col_comp = '#0072B2'

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
      label='Emissions + Durable CDR\n+ SLCP Mitigation',
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

  ax1.text(
      0.02,
      0.95,
      '(b)',
      transform=ax1.transAxes,
      fontsize=14,
      weight='normal',
      va='top',
      ha='left',
      zorder=10,
  )

  style_axes(ax2)

  # Base array for cumulative tracking
  years_full = np.arange(
      fair_utils.START_YEAR, fair_utils.START_YEAR + 1000
  )

  # Pre-calculate arrays to determine dynamic text offsets
  cdr_array = np.zeros(1000)
  cdr_indices = np.where(co2_cdr_adj < 0)[0]
  if len(cdr_indices) > 0:
    # Invert negative input to represent positive mitigation volume
    cdr_array[cdr_indices] = -co2_cdr_adj[cdr_indices]
  cdr_cum = np.cumsum(cdr_array)

  ch4_array = np.zeros(1000)
  if additional_ch4_req:
    for year, val in additional_ch4_req.items():
      idx = int(year - fair_utils.START_YEAR)
      if 0 <= idx < 1000:
        ch4_array[idx] += val * gwp_ch4 / 1000
  ch4_cum = np.cumsum(ch4_array)

  # Dynamically scale Y-axis based on cumulative totals
  max_y = max([np.max(cdr_cum), np.max(ch4_cum), 0.05]) * 1.15
  ax2.set_ylim(0, max_y)

  # Proportional offset for annotations
  y_offset = max_y * 0.08

  # 1. Cumulative CDR Line (Monochrome)
  if np.any(cdr_cum > 0):
    ax2.plot(
        years_full,
        cdr_cum,
        color='gray',
        linestyle='-',
        drawstyle='steps-post',
        linewidth=2.5,
        zorder=4,
    )

    # Place label pointing horizontally to the line from the right at year 2048
    idx_2045 = 2045 - fair_utils.START_YEAR
    if 0 <= idx_2045 < len(cdr_cum):
      val_2045 = cdr_cum[idx_2045]
      ax2.annotate(
          'Durable CDR',
          xy=(2046, val_2045),
          xytext=(2065, val_2045),
          arrowprops=dict(arrowstyle='->', color='black', lw=1.2, zorder=5),
          fontsize=12,
          va='center',
          ha='right',
          zorder=5,
      )

  # 2. Cumulative CH4 Mitigation Line (Monochrome)
  if np.any(ch4_cum > 0):
    ax2.plot(
        years_full,
        ch4_cum,
        color='gray',
        linestyle='-',
        drawstyle='steps-post',
        linewidth=2.5,
        zorder=4,
    )

    # Place label horizontally centered with a vertical arrow pointing to the
    # line at year 2030
    idx_2030 = 2030 - fair_utils.START_YEAR
    if 0 <= idx_2030 < len(ch4_cum):
      val_2030 = ch4_cum[idx_2030]
      ax2.annotate(
          'CH₄ Mitigation',
          xy=(2030, val_2030 + 0.02),
          xytext=(2040, val_2030 + 4 * y_offset),
          arrowprops=dict(arrowstyle='->', color='black', lw=1.2, zorder=5),
          fontsize=12,
          va='center',
          ha='right',
          zorder=5,
      )

  ax2.set_ylabel(
      f'Cumulative Mitigation\n(Gt CO₂e, GWP{gwp_horizon})',
      fontsize=12,
      weight='medium',
  )
  ax2.set_xlabel('Year', fontsize=12, weight='medium')
  ax2.tick_params(axis='both', which='major', labelsize=12)

  # Focus the x-axis around the relevant action window
  ax2.set_xlim(2025, 2100)

  plt.tight_layout()
  plt.subplots_adjust(hspace=0.08)

  final_save_path = os.path.join(
      fair_utils.TARGET_DIR, f'min_ch4_solver{use_solver}.png'
  )
  plt.savefig(final_save_path, dpi=300, bbox_inches='tight', facecolor='white')
  print(f'Figure saved to: {final_save_path}')

  plt.show()


if __name__ == '__main__':
  execute_calc_min_ch4(skip_calc=False)
