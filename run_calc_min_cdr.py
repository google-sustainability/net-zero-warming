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

"""Calculates minimum required CDR and plots compensation profiles."""

import os
import fair_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def execute_calc_min_cdr(
    co2_emission_adj=None,
    co2_cdr_adj=None,
    ch4_adj=None,
    hfc_adj=None,
    gwp_horizon=100,
    use_solver=True,
    tolerance=fair_utils.TOLERANCE,
    skip_calc=False,
):
  """Executes the CDR calculation and generates Figure 2.

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
    Tuple of (additional_cdr_required, df_uncompensated, df_compensated).
  """
  # Capture original input states before they are reassigned to default arrays
  ch4_provided = ch4_adj is not None
  hfc_provided = hfc_adj is not None

  if co2_emission_adj is None:
    co2_emission_adj = np.zeros(1000)
    co2_emission_adj[fair_utils.OFFSET] = 1.0

  if ch4_adj is None:
    ch4_adj = np.zeros(1000)
    ch4_adj[fair_utils.OFFSET] = -10

  if hfc_adj is None:
    hfc_adj = np.zeros(1000)

  if co2_cdr_adj is None:
    co2_cdr_adj = np.zeros(1000)

  if gwp_horizon == 100:
    gwp_ch4 = fair_utils.GWP100_CH4
    gwp_hfc = fair_utils.GWP100_HFC

  if gwp_horizon == 20:
    gwp_ch4 = fair_utils.GWP20_CH4
    gwp_hfc = fair_utils.GWP20_HFC

  uncomp_path = os.path.join(
      fair_utils.TARGET_DIR, 'temperature_uncompensated_min_cdr.csv'
  )
  comp_path = os.path.join(
      fair_utils.TARGET_DIR, 'temperature_compensated_min_cdr.csv'
  )
  cdr_schedule_path = os.path.join(fair_utils.TARGET_DIR, 'min_cdr.csv')

  if skip_calc:
    print('Skipping calculation. Loading previously generated outputs...')
    try:
      df_uncompensated = pd.read_csv(uncomp_path)
      df_compensated = pd.read_csv(comp_path)
      print('Loaded temperature profiles successfully.')
    except FileNotFoundError as exc:
      raise FileNotFoundError(
          'Could not find previous CSV outputs. Please run with skip_calc=False'
          ' first.'
      ) from exc

    # Load the CDR schedule if it exists
    if os.path.exists(cdr_schedule_path):
      df_cdr_schedule = pd.read_csv(cdr_schedule_path)
      additional_cdr_required = dict(
          zip(df_cdr_schedule['Year'], df_cdr_schedule['Required_CDR_Gt'])
      )
      print('Loaded CDR schedule successfully.')
    else:
      additional_cdr_required = {}

  else:
    print(f'Calculating Additional CDR (Solver Enabled: {use_solver})...')
    additional_cdr_required, df_uncompensated, df_compensated = (
        fair_utils.calc_min_cdr(
            co2_emission_adj=co2_emission_adj,
            co2_cdr_adj=co2_cdr_adj,
            ch4_adj=ch4_adj,
            hfc_adj=hfc_adj,
            use_solver=use_solver,
            tolerance=tolerance,
        )
    )

    df_uncompensated.to_csv(uncomp_path, index=False)
    print(f'\nUncompensated profile saved to: {uncomp_path}')

    df_compensated.to_csv(comp_path, index=False)
    print(f'Compensated profile saved to: {comp_path}')

    if additional_cdr_required:
      df_cdr_schedule = pd.DataFrame(
          list(additional_cdr_required.items()),
          columns=['Year', 'Required_CDR_Gt'],
      )
      df_cdr_schedule.to_csv(cdr_schedule_path, index=False)
      print(f'CDR Schedule saved to: {cdr_schedule_path}')
    else:
      # Clean up old file if the new run requires no CDR
      if os.path.exists(cdr_schedule_path):
        os.remove(cdr_schedule_path)

  # Use column 0 dynamically for FaIR indexing
  time_col = df_uncompensated.columns[0]

  # Scale the temperature values to 10^-3 °C for plotting
  for df in [df_uncompensated, df_compensated]:
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
      df_uncompensated[time_col],
      df_uncompensated['p50'],
      color=col_uncomp,
      label='Emissions + SLCP Mitigation',
      linewidth=2.5,
      zorder=3,
  )
  ax1.fill_between(
      df_uncompensated[time_col],
      df_uncompensated['p05'],
      df_uncompensated['p95'],
      color=col_uncomp,
      alpha=0.15,
      zorder=2,
  )

  ax1.plot(
      df_compensated[time_col],
      df_compensated['p50'],
      color=col_comp,
      label='Emissions + SLCP Mitigation + Durable CDR',
      linewidth=2.5,
      zorder=3,
  )
  ax1.fill_between(
      df_compensated[time_col],
      df_compensated['p05'],
      df_compensated['p95'],
      color=col_comp,
      alpha=0.15,
      zorder=2,
  )

  ax1.set_ylabel('Δ Temperature (10⁻³ °C)', fontsize=12, weight='medium')
  ax1.tick_params(axis='both', which='major', labelsize=12)

  # Move legend to the bottom right
  ax1.legend(frameon=False, fontsize=12, loc='lower right')

  # Add the (a) label to the top left
  ax1.text(
      0.02,
      0.95,
      '(a)',
      transform=ax1.transAxes,
      fontsize=14,
      weight='normal',
      va='top',
      ha='left',
      zorder=10,
  )

  # Adjusted limits for the 10^-3 scale
  ax1.set_ylim(-0.5, 0.5)

  style_axes(ax2)

  # Base array for cumulative tracking
  years_full = np.arange(
      fair_utils.START_YEAR, fair_utils.START_YEAR + 1000
  )

  # Pre-calculate arrays to determine dynamic text offsets
  ch4_array = np.zeros(1000)
  if ch4_provided and ch4_adj is not None:
    mask_ch4 = ch4_adj < 0
    ch4_array[mask_ch4] = -ch4_adj[mask_ch4] * gwp_ch4 / 1000
  ch4_cum = np.cumsum(ch4_array)

  hfc_array = np.zeros(1000)
  if hfc_provided and hfc_adj is not None:
    mask_hfc = hfc_adj < 0
    hfc_array[mask_hfc] = -hfc_adj[mask_hfc] * gwp_hfc / 1000000
  hfc_cum = np.cumsum(hfc_array)

  cdr_array = np.zeros(1000)
  if additional_cdr_required:
    for year, val in additional_cdr_required.items():
      idx = int(year - fair_utils.START_YEAR)
      if 0 <= idx < 1000:
        cdr_array[idx] += val
  cdr_cum = np.cumsum(cdr_array)

  # Dynamically scale Y-axis based on cumulative totals
  max_y = max([np.max(cdr_cum), np.max(ch4_cum), np.max(hfc_cum), 0.05]) * 1.15
  ax2.set_ylim(0, max_y)

  # Proportional offset for annotations
  y_offset = max_y * 0.08

  # 1. Cumulative CH4 Mitigation Line (Monochrome)
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
    max_val = np.max(ch4_cum)
    max_year = years_full[np.argmax(ch4_cum)]

    # Place label horizontally centered with a vertical arrow pointing to
    # the line
    text_x = min(max_year + 10, 2065)
    ax2.annotate(
        'CH₄ Mitigation',
        xy=(text_x, max_val - 0.02),
        xytext=(text_x, max_val - 3 * y_offset),
        arrowprops=dict(arrowstyle='->', color='black', lw=1.2, zorder=5),
        fontsize=12,
        va='top',
        ha='center',
        zorder=5,
    )

  # 2. Cumulative HFC Mitigation Line (Monochrome)
  if np.any(hfc_cum > 0):
    ax2.plot(
        years_full,
        hfc_cum,
        color='gray',
        linestyle='-',
        drawstyle='steps-post',
        linewidth=2.5,
        zorder=4,
    )
    max_val = np.max(hfc_cum)
    max_year = years_full[np.argmax(hfc_cum)]

    # Place label horizontally centered with a vertical arrow pointing to
    # the line
    text_x = min(max_year + 10, 2065)
    ax2.annotate(
        'HFC Mitigation',
        xy=(text_x, max_val - 0.02),
        xytext=(text_x, max_val + 4 * y_offset),
        arrowprops=dict(arrowstyle='->', color='black', lw=1.2, zorder=5),
        fontsize=12,
        va='bottom',
        ha='center',
        zorder=5,
    )

  # 3. Cumulative CDR Line (Monochrome)
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

    # Place label horizontally centered with a vertical arrow pointing to
    # the line near the end of the plot (year 2090)
    idx_2090 = 2090 - fair_utils.START_YEAR
    if 0 <= idx_2090 < len(cdr_cum):
      val_2090 = cdr_cum[idx_2090]
      ax2.annotate(
          'Durable CDR',
          xy=(2090, val_2090 - 0.02),
          xytext=(2090, val_2090 - 4 * y_offset),
          arrowprops=dict(arrowstyle='->', color='black', lw=1.2, zorder=5),
          fontsize=12,
          va='top',
          ha='center',
          zorder=5,
      )

  ax2.set_ylabel(
      f'Cumulative Mitigation\n(Gt CO₂e, GWP{gwp_horizon})',
      fontsize=12,
      weight='medium',
  )
  ax2.set_xlabel('Year', fontsize=12, weight='medium')
  ax2.tick_params(axis='both', which='major', labelsize=12)

  ax2.set_xlim(2025, 2100)

  plt.tight_layout()
  plt.subplots_adjust(hspace=0.08)

  final_save_path = os.path.join(
      fair_utils.TARGET_DIR, f'min_cdr_solver{use_solver}.png'
  )
  plt.savefig(final_save_path, dpi=300, bbox_inches='tight', facecolor='white')
  print(f'Figure saved to: {final_save_path}')

  plt.show()


if __name__ == '__main__':
  execute_calc_min_cdr(skip_calc=False)
