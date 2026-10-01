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

"""Generates Figure 3 for multi-gas mitigation scenarios."""

import os
import fair_utils
import matplotlib.pyplot as plt
import numpy as np


def execute_figure_3():
  """Generates Figure 3 demonstrating multi-gas mitigation portfolios."""
  # Example 1: 1 Gt CH4 mitigation in 2030, replaced by 1 Gt CDR in 2042.
  scenario_data_1 = {
      2030: {
          'CO2 emissions': 1e9,
          'Long-lived CDR': 0.0,
          'CH4 mitigation': 1.0e9,
      },
      2042: {
          'CO2 emissions': 0.0,
          'Long-lived CDR': 1.0e9,
          'CH4 mitigation': 0.0,
      },
  }

  # Example 2: Phased CH4 mitigation in 2026/2027, replaced by CDR in
  # 2037/2038/2039.
  scenario_data_2 = {
      2026: {
          'CO2 emissions': 0.0,
          'Long-lived CDR': 0.0,
          'CH4 mitigation': 0.4e9,
      },
      2027: {
          'CO2 emissions': 0.0,
          'Long-lived CDR': 0.0,
          'CH4 mitigation': 0.6e9,
      },
      2030: {
          'CO2 emissions': 1e9,
          'Long-lived CDR': 0.0,
          'CH4 mitigation': 0.0,
      },
      2037: {
          'CO2 emissions': 0.0,
          'Long-lived CDR': 0.2e9,
          'CH4 mitigation': 0.0,
      },
      2038: {
          'CO2 emissions': 0.0,
          'Long-lived CDR': 0.3e9,
          'CH4 mitigation': 0.0,
      },
      2039: {
          'CO2 emissions': 0.0,
          'Long-lived CDR': 0.5e9,
          'CH4 mitigation': 0.0,
      },
  }

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

  def process_scenario(scenario_data):
    years = sorted(list(scenario_data.keys()))
    co2_pulse = np.zeros(1000)
    co2_pulse_no_cdr = np.zeros(1000)
    ch4_pulse = np.zeros(1000)

    for year in years:
      idx = int(year) - fair_utils.START_YEAR
      if 0 <= idx < 1000:
        co2_tons = scenario_data[year].get('CO2 emissions', 0.0)
        cdr_tons = scenario_data[year].get('Long-lived CDR', 0.0)
        ch4_tons_gwp20 = scenario_data[year].get('CH4 mitigation', 0.0)

        co2_gt = co2_tons * 1e-9
        cdr_gt = cdr_tons * 1e-9
        ch4_mt = (ch4_tons_gwp20 * 1e-6) / fair_utils.GWP20_CH4

        co2_pulse[idx] += co2_gt
        co2_pulse[idx] -= cdr_gt
        co2_pulse_no_cdr[idx] += co2_gt
        ch4_pulse[idx] -= ch4_mt

    f_adj_co2 = fair_utils.run_fair(co2_adj=co2_pulse)
    t_adj_co2 = f_adj_co2.temperature.sel(
        scenario=fair_utils.SCENARIO, layer=0
    ).squeeze(drop=True)
    temp_diff_co2 = t_adj_co2 - t_base

    f_adj_em_ch4 = fair_utils.run_fair(
        co2_adj=co2_pulse_no_cdr, ch4_adj=ch4_pulse
    )
    t_adj_em_ch4 = f_adj_em_ch4.temperature.sel(
        scenario=fair_utils.SCENARIO, layer=0
    ).squeeze(drop=True)
    temp_diff_em_ch4 = t_adj_em_ch4 - t_base

    f_adj_all = fair_utils.run_fair(co2_adj=co2_pulse, ch4_adj=ch4_pulse)
    t_adj_all = f_adj_all.temperature.sel(
        scenario=fair_utils.SCENARIO, layer=0
    ).squeeze(drop=True)
    temp_diff_all = t_adj_all - t_base

    df_co2 = get_quantiles_df(temp_diff_co2)
    df_em_ch4 = get_quantiles_df(temp_diff_em_ch4)
    df_all = get_quantiles_df(temp_diff_all)

    schedule_years = np.arange(fair_utils.START_YEAR, 2101)
    ch4_arr = np.zeros_like(schedule_years, dtype=float)
    cdr_arr = np.zeros_like(schedule_years, dtype=float)

    for i, yr in enumerate(schedule_years):
      if yr in scenario_data:
        ch4_arr[i] = scenario_data[yr].get('CH4 mitigation', 0) / 1e9
        cdr_arr[i] = scenario_data[yr].get('Long-lived CDR', 0) / 1e9

    cum_ch4 = np.cumsum(ch4_arr)
    cum_cdr = np.cumsum(cdr_arr)

    return df_co2, df_em_ch4, df_all, schedule_years, cum_ch4, cum_cdr

  print('Processing Example 1...')
  res_1 = process_scenario(scenario_data_1)

  print('Processing Example 2...')
  res_2 = process_scenario(scenario_data_2)

  col_uncomp = '#E15759'  # Red
  col_ch4 = '#59A14F'  # Green
  col_comp = '#0072B2'  # Blue

  # 2x2 subplot layout with shared axes (equal height for all subplots)
  _, axes = plt.subplots(
      2,
      2,
      figsize=(14, 7),
      sharex=True,
      sharey='row',
      gridspec_kw={'height_ratios': [3, 1]},
  )

  ax1_1, ax1_2 = axes[0]  # Top row (Temperature)
  ax2_1, ax2_2 = axes[1]  # Bottom row (Cumulative schedules)

  def style_axes(ax_obj):
    ax_obj.spines['top'].set_visible(False)
    ax_obj.spines['right'].set_visible(False)
    ax_obj.tick_params(axis='both', which='major', labelsize=12)

  for ax in axes.flat:
    style_axes(ax)

  # Helper function to plot the temperature panels
  def plot_temperature_panel(ax, df_co2, df_em_ch4, df_all, panel_label):
    time_col = df_co2.columns[0]
    ax.axhline(0, color='gray', linestyle='-', linewidth=1.2, zorder=1)

    ax.plot(
        df_co2[time_col],
        df_co2['p50'] * 1000,
        color=col_uncomp,
        label='Emissions + Durable CDR',
        linewidth=2.5,
        zorder=3,
    )
    ax.fill_between(
        df_co2[time_col],
        df_co2['p05'] * 1000,
        df_co2['p95'] * 1000,
        color=col_uncomp,
        alpha=0.15,
        zorder=2,
    )

    ax.plot(
        df_em_ch4[time_col],
        df_em_ch4['p50'] * 1000,
        color=col_ch4,
        label='Emissions + CH₄ Mitigation',
        linewidth=2.5,
        zorder=4,
    )
    ax.fill_between(
        df_em_ch4[time_col],
        df_em_ch4['p05'] * 1000,
        df_em_ch4['p95'] * 1000,
        color=col_ch4,
        alpha=0.15,
        zorder=3,
    )

    ax.plot(
        df_all[time_col],
        df_all['p50'] * 1000,
        color=col_comp,
        label='Emissions + CH₄ Mitigation + Durable CDR',
        linewidth=2.5,
        zorder=5,
    )
    ax.fill_between(
        df_all[time_col],
        df_all['p05'] * 1000,
        df_all['p95'] * 1000,
        color=col_comp,
        alpha=0.15,
        zorder=4,
    )

    # Panel label (a)/(b) in the top-left corner, non-bold
    ax.text(
        0.02,
        0.95,
        panel_label,
        transform=ax.transAxes,
        fontsize=14,
        weight='normal',
        va='top',
        ha='left',
        zorder=10,
    )

  # Plot top row
  plot_temperature_panel(ax1_1, res_1[0], res_1[1], res_1[2], '(a)')
  plot_temperature_panel(ax1_2, res_2[0], res_2[1], res_2[2], '(b)')

  ax1_1.set_ylabel('Δ Temperature (10⁻³ °C)', fontsize=12, weight='medium')

  # Legend moved to ax1_1 (top left)
  ax1_1.legend(frameon=False, loc='lower right', fontsize=10)

  # Helper function to plot the cumulative schedule panels
  def plot_schedule_panel(
      ax, schedule_years, cum_ch4, cum_cdr, ch4_anno, cdr_anno, show_labels=True
  ):
    emission_yr = 2030
    emission_mass = 1.0  # 1 Gt CO2

    # Plot CO2 emission bar with the same color/alpha as the red shaded area
    ax.bar(
        emission_yr,
        emission_mass,
        width=0.5,
        color=col_uncomp,
        alpha=0.6,
        zorder=3,
    )

    # Plot mitigation lines with different opaqueness so overlaps are visible
    ax.step(
        schedule_years,
        cum_ch4,
        where='post',
        color='gray',
        linestyle='-',
        linewidth=2.5,
        alpha=0.4,
        zorder=4,
    )
    ax.step(
        schedule_years,
        cum_cdr,
        where='post',
        color='gray',
        linestyle='-',
        linewidth=2.5,
        alpha=1.0,
        zorder=5,
    )

    # Fill areas under the lines with distinct transparent grays
    ax.fill_between(
        schedule_years,
        0,
        cum_ch4,
        step='post',
        color='gray',
        alpha=0.1,
        zorder=2,
    )
    ax.fill_between(
        schedule_years,
        0,
        cum_cdr,
        step='post',
        color='gray',
        alpha=0.3,
        zorder=2,
    )

    ax.axhline(0, color='gray', linestyle='-', linewidth=1.2, zorder=1)

    # Only render labels and text if show_labels is True
    if show_labels:
      arrow_kwargs = dict(arrowstyle='->', color='black', lw=1.2)

      # Point to the left edge of the emission bar (2029.75) from the left
      ax.annotate(
          'CO₂\nEmission',
          xy=(2029.75, 0.25),
          xytext=(2028.5, 0.25),
          arrowprops=arrow_kwargs,
          fontsize=11,
          va='center',
          ha='right',
          zorder=10,
      )

      ax.annotate(
          'CH₄ Mitigation',
          xy=ch4_anno['xy'],
          xytext=ch4_anno['xytext'],
          arrowprops=arrow_kwargs,
          fontsize=11,
          va=ch4_anno['va'],
          ha=ch4_anno['ha'],
          zorder=10,
      )

      ax.annotate(
          'Durable CDR',
          xy=cdr_anno['xy'],
          xytext=cdr_anno['xytext'],
          arrowprops=arrow_kwargs,
          fontsize=11,
          va=cdr_anno['va'],
          ha=cdr_anno['ha'],
          zorder=10,
      )

  # Define specific annotation coordinates to point perfectly to the lines
  # for Example 1
  ch4_anno_1 = {
      'xy': (2035, 1.0),
      'xytext': (2035, 0.65),
      'va': 'top',
      'ha': 'center',
  }
  cdr_anno_1 = {
      'xy': (2042, 0.5),
      'xytext': (2044, 0.5),
      'va': 'center',
      'ha': 'left',
  }

  # Plot bottom left panel (with labels)
  plot_schedule_panel(
      ax2_1,
      res_1[3],
      res_1[4],
      res_1[5],
      ch4_anno_1,
      cdr_anno_1,
      show_labels=True,
  )

  # Define specific annotation coordinates for Example 2
  ch4_anno_2 = {
      'xy': (2033, 1.0),
      'xytext': (2033, 0.65),
      'va': 'top',
      'ha': 'center',
  }
  cdr_anno_2 = {
      'xy': (2038.5, 0.5),
      'xytext': (2041.5, 0.5),
      'va': 'center',
      'ha': 'left',
  }

  # Plot bottom right panel (without labels)
  plot_schedule_panel(
      ax2_2,
      res_2[3],
      res_2[4],
      res_2[5],
      ch4_anno_2,
      cdr_anno_2,
      show_labels=False,
  )

  ax2_1.set_xlabel('Year', fontsize=12, weight='medium')
  ax2_2.set_xlabel('Year', fontsize=12, weight='medium')
  ax2_1.set_ylabel(
      'Cumulative Mitigation\n(Gt CO₂e, GWP20)', fontsize=12, weight='medium'
  )

  # Set limits
  for ax in axes.flat:
    ax.set_xlim(2025, 2050)
  ax2_1.set_ylim(0, 1.1)

  # Tidy layout
  plt.tight_layout()
  plt.subplots_adjust(hspace=0.08, wspace=0.1)

  # Save logic
  os.makedirs(fair_utils.TARGET_DIR, exist_ok=True)
  filename = 'Figure_3.png'
  full_path = os.path.join(fair_utils.TARGET_DIR, filename)
  plt.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
  print(f'Figure successfully saved to: {full_path}')


if __name__ == '__main__':
  execute_figure_3()
