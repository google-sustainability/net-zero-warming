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

"""Generates Figure 1 comparing CO2 and CH4 temperature impacts."""

import os
import fair_utils
import matplotlib.pyplot as plt
import numpy as np


def execute_figure_1(co2_pulse_mag=1.0, ch4_pulse_mag=None):
  """Executes the initial CH₄ vs CO₂ comparison and generates Figure 1.

  Args:
    co2_pulse_mag: Magnitude of the CO2 pulse.
    ch4_pulse_mag: Magnitude of the CH4 pulse.

  Returns:
    The year where temperature difference crosses zero, or None.
  """
  if ch4_pulse_mag is None:
    ch4_pulse_mag = -1.0 * 1e3 / fair_utils.GWP20_CH4

  co2_pulse = np.zeros(1000)
  co2_pulse[fair_utils.OFFSET] = co2_pulse_mag

  ch4_pulse = np.zeros(1000)
  ch4_pulse[fair_utils.OFFSET] = ch4_pulse_mag

  print('Running Baseline...')
  f_base = fair_utils.run_fair()

  print('Running Adjusted Scenario (In-Memory)...')
  f_adj = fair_utils.run_fair(co2_adj=co2_pulse, ch4_adj=ch4_pulse)

  # Calculate temperature difference
  t_base = f_base.temperature.sel(
      scenario=fair_utils.SCENARIO, layer=0
  ).squeeze(drop=True)
  t_adj = f_adj.temperature.sel(scenario=fair_utils.SCENARIO, layer=0).squeeze(
      drop=True
  )
  temp_diff = t_adj - t_base

  # Calculate quantiles and convert directly to a DataFrame in memory
  percentiles_to_calc = sorted(
      list(set([0.05, 0.50, 0.95, fair_utils.EVAL_PERCENTILE]))
  )
  q = temp_diff.quantile(percentiles_to_calc, dim='config')
  time_dim = [d for d in q.dims if d != 'quantile'][0]
  df = q.transpose(time_dim, 'quantile').to_pandas()
  df.columns = [f'p{int(col * 100):02d}' for col in df.columns]
  df = df.reset_index()

  # Use column 0 dynamically for FaIR indexing
  time_col = df.columns[0]

  # Scale the temperature values to 10^-3 °C
  for col in df.columns:
    if col != time_col:
      df[col] *= 1000

  col_all = '#E15759'
  _, ax1 = plt.subplots(figsize=(8, 5))

  ax1.spines['top'].set_visible(False)
  ax1.spines['right'].set_visible(False)

  ax1.axhline(0, color='gray', linestyle='-', linewidth=1.2, zorder=1)
  ax1.plot(
      df[time_col],
      df['p50'],
      color=col_all,
      label='CO₂ Emissions + CH₄ Mitigation',
      linewidth=2.5,
      zorder=3,
  )
  ax1.fill_between(
      df[time_col], df['p05'], df['p95'], color=col_all, alpha=0.15, zorder=2
  )

  diffs = df['p50'].values
  times = df[time_col].values

  # Only look for zero-crossings strictly after the intervention year
  post_pulse_mask = times > fair_utils.TARGET_YEAR
  valid_indices = np.where(post_pulse_mask & (diffs > 0))[0]

  crossing_year = None
  if len(valid_indices) > 0:
    cross_idx = valid_indices[0]
    if cross_idx > 0:
      x0, x1 = times[cross_idx - 1], times[cross_idx]
      y0, y1 = diffs[cross_idx - 1], diffs[cross_idx]
      crossing_year = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)

  end_year = 2100
  y_max = df['p95'].max()

  # Establish base Y heights for the arrows
  arrow_y_top = y_max * 0.4
  arrow_y_bottom = y_max * -0.3

  if crossing_year:
    # - Interval 1: Intervention to Crossing -
    # Draw double-headed arrow
    ax1.annotate(
        '',
        xy=(fair_utils.TARGET_YEAR, arrow_y_top),
        xytext=(crossing_year, arrow_y_top),
        arrowprops=dict(
            arrowstyle='<-', color='gray', shrinkA=0, shrinkB=0, linewidth=1.2
        ),
    )

    # Place text slightly above the center of the arrow
    ax1.text(
        (fair_utils.TARGET_YEAR + crossing_year) / 2,
        arrow_y_top + (y_max * 0.05),
        'Cooling by CH₄\nmitigation exceeds\nwarming by\nCO₂ emission',
        ha='center',
        va='bottom',
        fontsize=10,
        color='black',
    )

    # - Interval 2: Crossing to End Year -
    # Draw double-headed arrow
    ax1.annotate(
        '',
        xy=(crossing_year, arrow_y_bottom),
        xytext=(end_year, arrow_y_bottom),
        arrowprops=dict(
            arrowstyle='->', color='gray', shrinkA=0, shrinkB=0, linewidth=1.2
        ),
    )

    # Place text slightly below the center of the arrow
    ax1.text(
        (crossing_year + end_year) / 2,
        arrow_y_bottom - (y_max * 0.05),
        'Warming by CO₂ emission exceeds cooling by CH₄ \nmitigation:'
        ' additional GHG mitigation required',
        ha='center',
        va='top',
        fontsize=10,
        color='black',
    )

    # Draw the vertical line at the crossing point
    ax1.axvline(
        crossing_year, color='gray', linestyle='-', linewidth=1.2, zorder=1
    )

  else:
    # Fallback if no crossing occurs
    ax1.text(
        2046,
        arrow_y_top,
        'Cooling by CH₄ \nmitigation exceeds \nwarming by CO₂ \nemission',
        ha='center',
        va='bottom',
        fontsize=10,
        color='black',
    )

  ax1.set_ylabel('Δ Temperature (10⁻³ °C)', fontsize=12, weight='medium')
  ax1.set_xlabel('Year', fontsize=12, weight='medium')
  ax1.tick_params(axis='both', which='major', labelsize=10)
  ax1.set_xlim(2025, end_year)
  ax1.legend(frameon=False, loc='upper right')

  # Pad the y-axis to ensure text doesn't get cut off
  current_ymin, current_ymax = ax1.get_ylim()
  ax1.set_ylim(
      min(current_ymin, arrow_y_bottom - (y_max * 0.3)),
      max(current_ymax, arrow_y_top + (y_max * 0.05)),
  )

  plt.tight_layout()

  full_path = os.path.join(fair_utils.TARGET_DIR, 'Figure_1.png')
  plt.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
  print(f'Figure saved to: {full_path}')

  plt.show()
  print(f'Crossing year is: {crossing_year}')

  return crossing_year


if __name__ == '__main__':
  execute_figure_1()
