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

"""Generates Figure 4 demonstrating stacked mitigation schedules."""

import os
import fair_utils
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np


def execute_figure_4():
  """Generates Figure 4 demonstrating stacked mitigation schedules."""

  # Format: (Year, Mass in Gt CO2)
  emissions = [
      (2030, 1.0),
      (2031, 1.0),
      (2032, 0.5),
  ]

  # Format: (Issuance Year, Gas Type, Mass in Gt CO2e at GWP20)
  mitigation_issuances = [
      (2028, 'CH4', 0.2),
      (2028, 'HFC-134a', 0.1),
      (2029, 'CH4', 0.2),
      (2029, 'HFC-134a', 0.1),
      (2030, 'CH4', 0.2),
      (2030, 'HFC-134a', 0.2),
      # Neutralizing the 2031 emission addition
      (2031, 'CH4', 0.5),
      (2031, 'HFC-134a', 0.5),
      # Neutralizing the 2032 emission addition
      (2032, 'CH4', 0.3),
      (2032, 'HFC-134a', 0.2),
  ]

  # Replacement Rules derived from Table 2
  replacement_delays = {
      'CH4': 12,  # CH4 must be replaced at latest 12 years after issuance
      'HFC-134a': (
          10  # HFC-134a must be replaced at latest 11 years after issuance
      ),
  }

  co2_pulse = np.zeros(1000)
  co2_pulse_no_cdr = np.zeros(1000)
  ch4_pulse = np.zeros(1000)
  hfc_pulse = np.zeros(1000)

  # Apply Emissions
  for em_yr, mass in emissions:
    idx = em_yr - fair_utils.START_YEAR
    if 0 <= idx < 1000:
      co2_pulse[idx] += mass
      co2_pulse_no_cdr[idx] += mass

  # Apply Mitigation and CDR Replacements
  for issue_yr, gas_type, mass in mitigation_issuances:
    delay = replacement_delays.get(gas_type, 0)
    replace_yr = issue_yr + delay

    issue_idx = issue_yr - fair_utils.START_YEAR
    replace_idx = replace_yr - fair_utils.START_YEAR

    # SLCP issuance (negative pulse)
    if 0 <= issue_idx < 1000:
      if gas_type == 'CH4':
        # Convert Gt CO2e to Mt CH4
        ch4_pulse[issue_idx] -= (mass * 1000.0) / fair_utils.GWP20_CH4
      elif gas_type == 'HFC-134a':
        # Convert Gt CO2e to kt HFC-134a
        hfc_pulse[issue_idx] -= (mass * 1000000.0) / fair_utils.GWP20_HFC

    # Durable CDR replacement (negative CO2 pulse)
    if 0 <= replace_idx < 1000:
      co2_pulse[replace_idx] -= mass

  print('Running Baseline...')
  f_base = fair_utils.run_fair()
  t_base = f_base.temperature.sel(
      scenario=fair_utils.SCENARIO, layer=0
  ).squeeze(drop=True)

  def get_quantiles_df(temp_diff):
    percentiles_to_calc = sorted(
        list(set([0.05, 0.50, 0.95, fair_utils.EVAL_PERCENTILE]))
    )
    q = temp_diff.quantile(percentiles_to_calc, dim='config')
    time_dim = [d for d in q.dims if d != 'quantile'][0]
    df_q = q.transpose(time_dim, 'quantile').to_pandas()
    df_q.columns = [f'p{int(col * 100):02d}' for col in df_q.columns]
    return df_q.reset_index()

  print('Running Emissions + Durable CDR...')
  f_cdr = fair_utils.run_fair(co2_adj=co2_pulse)
  t_cdr = f_cdr.temperature.sel(
      scenario=fair_utils.SCENARIO, layer=0
  ).squeeze(drop=True)
  df_cdr = get_quantiles_df(t_cdr - t_base)

  print('Running Emissions + SLCP Mitigation...')
  f_slcp = fair_utils.run_fair(
      co2_adj=co2_pulse_no_cdr, ch4_adj=ch4_pulse, hfc_adj=hfc_pulse
  )
  t_slcp = f_slcp.temperature.sel(
      scenario=fair_utils.SCENARIO, layer=0
  ).squeeze(drop=True)
  df_slcp = get_quantiles_df(t_slcp - t_base)

  print('Running Emissions + SLCP Mitigation + Durable CDR...')
  f_all = fair_utils.run_fair(
      co2_adj=co2_pulse, ch4_adj=ch4_pulse, hfc_adj=hfc_pulse
  )
  t_all = f_all.temperature.sel(
      scenario=fair_utils.SCENARIO, layer=0
  ).squeeze(drop=True)
  df_all = get_quantiles_df(t_all - t_base)

  years_stack = np.arange(2025, 2060)
  cum_emissions = np.zeros_like(years_stack, dtype=float)
  ch4_active = np.zeros_like(years_stack, dtype=float)
  hfc_active = np.zeros_like(years_stack, dtype=float)
  cdr_active = np.zeros_like(years_stack, dtype=float)

  for em_yr, em_mass in emissions:
    for i, yr in enumerate(years_stack):
      if yr >= em_yr:
        cum_emissions[i] += em_mass

  for issue_yr, gas_type, mass in mitigation_issuances:
    delay = replacement_delays.get(gas_type, 0)
    replace_yr = issue_yr + delay

    for i, yr in enumerate(years_stack):
      if issue_yr <= yr < replace_yr:
        if gas_type == 'CH4':
          ch4_active[i] += mass
        elif gas_type == 'HFC-134a':
          hfc_active[i] += mass
      elif yr >= replace_yr:
        cdr_active[i] += mass

  stack_1 = ch4_active
  stack_2 = stack_1 + hfc_active
  stack_3 = stack_2 + cdr_active

  col_uncomp = '#E15759'  # Red
  col_ch4 = '#59A14F'  # Green
  col_hfc = '#9467BD'  # Purple
  col_comp = '#0072B2'  # Blue

  # Consistent aspect ratio and size with other 2-panel vertical plots
  _, (ax1, ax2) = plt.subplots(
      2, 1, figsize=(8, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]}
  )

  def style_axes(ax_obj):
    ax_obj.spines['top'].set_visible(False)
    ax_obj.spines['right'].set_visible(False)
    ax_obj.tick_params(axis='y', which='major', labelsize=12)

  style_axes(ax1)
  style_axes(ax2)

  time_col = df_cdr.columns[0]
  ax1.axhline(0, color='gray', linestyle='-', linewidth=1.2, zorder=1)

  # Red: Emissions + CDR
  ax1.plot(
      df_cdr[time_col],
      df_cdr['p50'] * 1000,
      color=col_uncomp,
      label='Emissions + Durable CDR',
      linewidth=2.5,
      zorder=3,
  )
  ax1.fill_between(
      df_cdr[time_col],
      df_cdr['p05'] * 1000,
      df_cdr['p95'] * 1000,
      color=col_uncomp,
      alpha=0.15,
      zorder=2,
  )

  # Green: Emissions + SLCP
  ax1.plot(
      df_slcp[time_col],
      df_slcp['p50'] * 1000,
      color=col_ch4,
      label='Emissions + SLCP Mitigation',
      linewidth=2.5,
      zorder=4,
  )
  ax1.fill_between(
      df_slcp[time_col],
      df_slcp['p05'] * 1000,
      df_slcp['p95'] * 1000,
      color=col_ch4,
      alpha=0.15,
      zorder=3,
  )

  # Blue: Emissions + SLCP + CDR
  ax1.plot(
      df_all[time_col],
      df_all['p50'] * 1000,
      color=col_comp,
      label='Emissions + SLCP Mitigation\n+ Durable CDR',
      linewidth=2.5,
      zorder=5,
  )
  ax1.fill_between(
      df_all[time_col],
      df_all['p05'] * 1000,
      df_all['p95'] * 1000,
      color=col_comp,
      alpha=0.15,
      zorder=4,
  )

  ax1.set_ylabel('Δ Temperature (10⁻³ °C)', fontsize=12, weight='medium')

  # Ensure there is enough vertical space for the legend
  top_y = max(df_cdr['p95'].max() * 1000, df_all['p95'].max() * 1000)
  ax1.set_ylim(
      bottom=ax1.get_ylim()[0], top=max(top_y * 1.5, ax1.get_ylim()[1])
  )

  ax1.legend(frameon=False, loc='upper right', fontsize=10)
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

  ax2.axhline(0, color='gray', linestyle='-', linewidth=1.2, zorder=1)

  # Target Line
  ax2.step(
      years_stack,
      cum_emissions,
      where='post',
      color='gray',
      linestyle='-',
      linewidth=2.5,
      zorder=7,
      label='Cumulative CO₂ Emissions',
  )

  # Shaded Portfolio (no solid outlines, formatted like temperature
  # fill_between)
  ax2.fill_between(
      years_stack,
      0,
      stack_1,
      step='post',
      color=col_ch4,
      alpha=0.15,
      label='CH₄ Mitigation',
      zorder=2,
  )
  ax2.fill_between(
      years_stack,
      stack_1,
      stack_2,
      step='post',
      color=col_hfc,
      alpha=0.15,
      label='HFC-134a Mitigation',
      zorder=2,
  )
  ax2.fill_between(
      years_stack,
      stack_2,
      stack_3,
      step='post',
      color=col_comp,
      alpha=0.15,
      label='Durable CDR',
      zorder=2,
  )

  ax2.set_ylabel(
      'Emissions and Mitigation\n(Gt CO₂e, GWP20)', fontsize=12, weight='medium'
  )
  ax2.set_xlabel('Year', fontsize=12, weight='medium')

  ax2.legend(frameon=False, loc='lower right', fontsize=10, ncol=1)
  ax2.text(
      0.02,
      0.95,
      '(b)',
      transform=ax2.transAxes,
      fontsize=14,
      weight='normal',
      va='top',
      ha='left',
      zorder=10,
  )

  # Focus the x-axis around the active mitigation window
  ax2.set_xlim(2025, 2055)
  ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
  ax2.tick_params(axis='x', labelsize=11, rotation=0)

  # Adjust Y limits dynamically with a higher multiplier (1.8) to leave room
  # for the legend in the shorter panel
  max_y = max(np.max(cum_emissions), np.max(stack_3)) * 1.1
  ax2.set_ylim(0, max_y)

  plt.tight_layout()
  plt.subplots_adjust(hspace=0.08)

  # Save logic
  os.makedirs(fair_utils.TARGET_DIR, exist_ok=True)
  filename = 'Figure_4.png'
  full_path = os.path.join(fair_utils.TARGET_DIR, filename)
  plt.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
  print(f'Figure successfully saved to: {full_path}')
  plt.show()


if __name__ == '__main__':
  execute_figure_4()
