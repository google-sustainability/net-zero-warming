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

"""Simulates dynamic portfolio adaptation to CDR delays and scientific updates.

Generates:
  1. Supplementary Table S5 (via run_calibration_sensitivity)
  2. Figure 4a trajectories (5-year durable CDR delivery delay from 2042 to
     2047, comparing no adaptation with optimized and standardized CH4 bridges)
  3. Figure 4b trajectories (2037 reassessment under a 15% shorter unperturbed
     CH4 lifetime, comparing no adaptation with advancing CDR or adding CH4)
"""

import os

import fair_utils
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import run_calibration_sensitivity


def _extract_quantiles_df(t_scen, t_base, config_key, trajectory_type):
  """Extracts p05, p50, and p95 delta-T trajectories into a tidy DataFrame."""
  diff = t_scen - t_base
  q = diff.quantile([0.05, 0.50, 0.95], dim='config')
  time_dim = [d for d in q.dims if d != 'quantile'][0]
  df = q.transpose(time_dim, 'quantile').to_pandas()
  df.columns = ['p05', 'p50', 'p95']
  df = df.reset_index().rename(columns={time_dim: 'year'})
  df['config_key'] = config_key
  df['trajectory'] = trajectory_type
  return df


def _find_zero_crossing_year(years_arr, temp_arr, after_year):
  """Finds linearly interpolated continuous zero-crossing year after after_year."""
  mask = years_arr > after_year
  yrs = years_arr[mask]
  vals = temp_arr[mask]
  pos_idx = np.where(vals > 0.0)[0]
  if len(pos_idx) == 0:
    return None, None
  k = int(pos_idx[0])
  first_pos_timebound = int(round(yrs[k]))
  if k == 0:
    return float(yrs[k]), first_pos_timebound
  y0, y1 = float(yrs[k - 1]), float(yrs[k])
  v0, v1 = float(vals[k - 1]), float(vals[k])
  cont_year = y0 + (0.0 - v0) * (y1 - y0) / (v1 - v0) if (v1 != v0) else y1
  return cont_year, first_pos_timebound


def _solve_midcourse_ch4_topup(
    cfg,
    t_base,
    co2_sched,
    base_ch4_sched,
    reassessment_year,
    scheduled_cdr_year,
    eval_percentile=fair_utils.EVAL_PERCENTILE,
    search_high=25.0,
    tolerance=fair_utils.TOLERANCE,
    sim_max_year=2100,
):
  """Solves for minimum remedial CH4 mitigation pulse in reassessment_year.

  Uses binary search to find the smallest CH4 pulse (in Mt CH4) issued in
  reassessment_year that holds delta_T (at eval_percentile) <= 0 C across
  [reassessment_year + 1, scheduled_cdr_year].

  Args:
    cfg: Configuration dictionary from ensure_calibration_files.
    t_base: Baseline surface temperature DataArray for cfg.
    co2_sched: Array of CO2 emission and CDR adjustments.
    base_ch4_sched: Array of existing CH4 mitigation adjustments.
    reassessment_year: Year in which the remedial CH4 bridge is issued.
    scheduled_cdr_year: Year in which durable CDR is delivered.
    eval_percentile: Ensemble quantile evaluated for warming neutralization.
    search_high: Upper bound for binary search (Mt CH4).
    tolerance: Convergence tolerance (Mt CH4).
    sim_max_year: End year of the FaIR simulation.

  Returns:
    Tuple of (required_topup_ch4_mt, iterations, final_fair_instance).
  """
  idx_start = (reassessment_year + 1) - fair_utils.MIN_YEAR
  idx_end = scheduled_cdr_year - fair_utils.MIN_YEAR
  idx_reassess_emis = reassessment_year - fair_utils.START_YEAR

  f_zero = run_calibration_sensitivity.run_fair_config(
      cfg, co2_adj=co2_sched, ch4_adj=base_ch4_sched, max_year=sim_max_year
  )
  t_zero = f_zero.temperature.sel(
      scenario=fair_utils.SCENARIO, layer=0
  ).squeeze(drop=True)
  q_zero = (t_zero - t_base).quantile(eval_percentile, dim='config')
  max_anom_zero = float(q_zero[idx_start : idx_end + 1].max().values)
  if max_anom_zero <= 0.0:
    return 0.0, 0, f_zero

  low, high = 0.0, search_high
  iteration = 0
  while (high - low) > tolerance and iteration < 25:
    iteration += 1
    mid = (low + high) / 2.0
    ch4_test = base_ch4_sched.copy()
    ch4_test[idx_reassess_emis] -= mid

    f_test = run_calibration_sensitivity.run_fair_config(
        cfg, co2_adj=co2_sched, ch4_adj=ch4_test, max_year=sim_max_year
    )
    t_test = f_test.temperature.sel(
        scenario=fair_utils.SCENARIO, layer=0
    ).squeeze(drop=True)
    q_test = (t_test - t_base).quantile(eval_percentile, dim='config')
    max_anom = float(q_test[idx_start : idx_end + 1].max().values)

    if max_anom > 0.0:
      low = mid
    else:
      high = mid

  ch4_final = base_ch4_sched.copy()
  ch4_final[idx_reassess_emis] -= high
  f_final = run_calibration_sensitivity.run_fair_config(
      cfg, co2_adj=co2_sched, ch4_adj=ch4_final, max_year=sim_max_year
  )
  return high, iteration, f_final


def run_cdr_delay_simulation(
    configs,
    df_s5_raw=None,
    emission_co2_gt=1.0,
    initial_ch4_mt=None,
    scheduled_cdr_year=2042,
    cdr_delay_years=5,
    sim_max_year=2100,
):
  """Simulates portfolio adaptation to a durable CDR delivery delay (Figure 4a).

  Evaluates a 2030 portfolio (1 Gt CO2 + 1 Gt CO2e GWP20 CH4) when durable CDR
  is delayed from scheduled_cdr_year (2042) by cdr_delay_years (to 2047) under:
    1. No adaptation (CDR delayed to 2047 without an extension bridge)
    2. Standardized bridge from Table S1 added in 2042
    3. Model-optimized bridge solved directly via FaIR added in 2042

  Args:
    configs: Dictionary of FaIR configurations from ensure_calibration_files.
    df_s5_raw: Optional raw Table S5 DataFrame used to look up standardized
      CH4 ratios without re-solving.
    emission_co2_gt: Magnitude of the CO2 emission pulse in Gt CO2.
    initial_ch4_mt: Initial CH4 mitigation pulse in Mt CH4 (defaults to 1:1
      CO2e at GWP20).
    scheduled_cdr_year: Originally contracted durable CDR delivery year.
    cdr_delay_years: Length of the delivery delay in years.
    sim_max_year: End year of the FaIR simulation.

  Returns:
    Tuple of (df_delay_summary, df_delay_traj).
  """
  base_cfg = configs['baseline_v1.4.1_ch4mod']
  cfg_key = base_cfg['key']
  delayed_cdr_year = scheduled_cdr_year + cdr_delay_years

  if initial_ch4_mt is None:
    initial_ch4_mt = (emission_co2_gt * 1000.0) / fair_utils.GWP20_CH4

  print(
      f'\nSimulating {cdr_delay_years}-year durable CDR delivery delay'
      f" ({scheduled_cdr_year} -> {delayed_cdr_year}) under {base_cfg['label']}..."
  )
  f_base = run_calibration_sensitivity.run_fair_config(
      base_cfg, max_year=sim_max_year, progress=True
  )
  t_base = f_base.temperature.sel(
      scenario=fair_utils.SCENARIO, layer=0
  ).squeeze(drop=True)

  std_bridge_per_gt = None
  if df_s5_raw is not None:
    match = df_s5_raw[
        (df_s5_raw['Config_Key'] == cfg_key)
        & (df_s5_raw['Delay_Years'] == cdr_delay_years)
        & (df_s5_raw['Percentile'] == 95)
    ]
    if not match.empty:
      std_bridge_per_gt = float(match['Required_CH4_Mt_per_Gt_CO2'].values[0])

  if std_bridge_per_gt is None:
    std_bridge_per_gt, _ = (
        run_calibration_sensitivity.calc_min_ch4_delay_for_config(
            cfg=base_cfg,
            delay_years=cdr_delay_years,
            t_base=t_base,
            eval_percentile=fair_utils.EVAL_PERCENTILE,
            max_year=sim_max_year,
        )
    )

  std_bridge_mt = std_bridge_per_gt * emission_co2_gt
  std_bridge_gwp20 = (std_bridge_mt * fair_utils.GWP20_CH4) / 1000.0

  idx_emis = fair_utils.TARGET_YEAR - fair_utils.START_YEAR
  idx_sched_cdr = scheduled_cdr_year - fair_utils.START_YEAR
  idx_delayed_cdr = delayed_cdr_year - fair_utils.START_YEAR

  co2_delayed = np.zeros(1000)
  co2_delayed[idx_emis] = emission_co2_gt
  co2_delayed[idx_delayed_cdr] = -emission_co2_gt

  ch4_unadj = np.zeros(1000)
  ch4_unadj[idx_emis] = -initial_ch4_mt

  traj_dfs = []

  # 1. No adaptation (CDR delayed to 2047 without an extension bridge)
  f_unadj = run_calibration_sensitivity.run_fair_config(
      base_cfg, co2_adj=co2_delayed, ch4_adj=ch4_unadj, max_year=sim_max_year
  )
  t_unadj = f_unadj.temperature.sel(
      scenario=fair_utils.SCENARIO, layer=0
  ).squeeze(drop=True)
  df_unadj = _extract_quantiles_df(
      t_unadj, t_base, cfg_key, 'unadjusted_delayed_cdr'
  )
  traj_dfs.append(df_unadj)

  years_arr = df_unadj['year'].values
  cont_cross_p95, _ = _find_zero_crossing_year(
      years_arr, df_unadj['p95'].values, after_year=fair_utils.TARGET_YEAR + 1
  )
  cont_cross_p50, _ = _find_zero_crossing_year(
      years_arr, df_unadj['p50'].values, after_year=fair_utils.TARGET_YEAR + 1
  )

  delay_window_mask = (years_arr >= scheduled_cdr_year) & (
      years_arr <= delayed_cdr_year
  )
  max_unadj_p95 = float(df_unadj.loc[delay_window_mask, 'p95'].max())
  max_unadj_p50 = float(df_unadj.loc[delay_window_mask, 'p50'].max())

  # 2. Adapted (standardized): Table S1 CH4 bridge added in 2042
  ch4_std_bridge = ch4_unadj.copy()
  ch4_std_bridge[idx_sched_cdr] -= std_bridge_mt
  f_std = run_calibration_sensitivity.run_fair_config(
      base_cfg,
      co2_adj=co2_delayed,
      ch4_adj=ch4_std_bridge,
      max_year=sim_max_year,
  )
  t_std = f_std.temperature.sel(scenario=fair_utils.SCENARIO, layer=0).squeeze(
      drop=True
  )
  df_std = _extract_quantiles_df(
      t_std, t_base, cfg_key, 'adapted_std_ch4_bridge'
  )
  traj_dfs.append(df_std)
  max_std_p95 = float(df_std.loc[delay_window_mask, 'p95'].max())

  # 3. Adapted (optimized): Solver-minimized CH4 bridge added in 2042
  opt_bridge_mt, opt_iters, f_opt = _solve_midcourse_ch4_topup(
      cfg=base_cfg,
      t_base=t_base,
      co2_sched=co2_delayed,
      base_ch4_sched=ch4_unadj,
      reassessment_year=scheduled_cdr_year,
      scheduled_cdr_year=delayed_cdr_year,
      eval_percentile=fair_utils.EVAL_PERCENTILE,
      tolerance=fair_utils.TOLERANCE,
      sim_max_year=sim_max_year,
  )
  t_opt = f_opt.temperature.sel(scenario=fair_utils.SCENARIO, layer=0).squeeze(
      drop=True
  )
  df_opt = _extract_quantiles_df(
      t_opt, t_base, cfg_key, 'adapted_opt_ch4_bridge'
  )
  traj_dfs.append(df_opt)
  opt_bridge_gwp20 = (opt_bridge_mt * fair_utils.GWP20_CH4) / 1000.0

  print(
      f'  -> No adaptation ({delayed_cdr_year} CDR): p95 crosses 0 C at'
      f' {cont_cross_p95:.2f} (p50 crosses at {cont_cross_p50:.2f});'
      f' peak p95 = {max_unadj_p95 * 1000.0:+.3f} mK'
  )
  print(
      f'  -> Standardized {cdr_delay_years}-yr CH4 bridge in'
      f' {scheduled_cdr_year}: +{std_bridge_mt:.2f} Mt CH4'
      f' ({std_bridge_gwp20:.2f} Gt CO2e at GWP20);'
      f' peak p95 across delay = {max_std_p95 * 1000.0:+.3f} mK'
  )
  print(
      f'  -> Model-optimized CH4 bridge in {scheduled_cdr_year}:'
      f' +{opt_bridge_mt:.2f} Mt CH4 ({opt_bridge_gwp20:.2f} Gt CO2e at GWP20)'
  )

  df_delay_summary = pd.DataFrame([{
      'Config_Key': cfg_key,
      'Config_Label': base_cfg['label'],
      'Emission_Year': fair_utils.TARGET_YEAR,
      'Scheduled_CDR_Year': scheduled_cdr_year,
      'CDR_Delay_Years': cdr_delay_years,
      'Delayed_CDR_Year': delayed_cdr_year,
      'Initial_2030_CH4_Mt': initial_ch4_mt,
      'Unadjusted_Crossing_Year_p95': cont_cross_p95,
      'Unadjusted_Crossing_Year_p50': cont_cross_p50,
      'Max_Unadjusted_Delay_dT_p95_degC': max_unadj_p95,
      'Max_Unadjusted_Delay_dT_p50_degC': max_unadj_p50,
      'Standardized_Bridge_CH4_Mt': std_bridge_mt,
      'Standardized_Bridge_GWP20_tCO2e_per_tCO2': (
          std_bridge_gwp20 / emission_co2_gt
      ),
      'Optimized_Bridge_CH4_Mt': opt_bridge_mt,
      'Optimized_Bridge_GWP20_tCO2e_per_tCO2': (
          opt_bridge_gwp20 / emission_co2_gt
      ),
      'Solver_Iterations': opt_iters,
  }])
  df_delay_traj = pd.concat(traj_dfs, ignore_index=True)
  return df_delay_summary, df_delay_traj


def run_midcourse_simulation(
    configs,
    emission_co2_gt=1.0,
    initial_ch4_mt=None,
    scheduled_cdr_year=2042,
    reassessment_year=2037,
    sim_max_year=2100,
):
  """Simulates portfolio re-evaluation under updated configurations (Figure 4b).

  Evaluates a baseline-sized 2030-to-2042 portfolio across all four FaIR
  configurations and solves for mid-course adaptations in reassessment_year
  (2037) when p95 exceeds 0 C prior to 2042 CDR delivery.

  Args:
    configs: Dictionary of FaIR configurations from ensure_calibration_files.
    emission_co2_gt: Magnitude of the CO2 emission pulse in Gt CO2.
    initial_ch4_mt: Initial CH4 mitigation pulse in Mt CH4 (defaults to 1:1
      CO2e at GWP20).
    scheduled_cdr_year: Originally contracted durable CDR delivery year.
    reassessment_year: Year in which scientific reassessment occurs.
    sim_max_year: End year of the FaIR simulation.

  Returns:
    Tuple of (df_mid_summary, df_mid_traj).
  """
  if initial_ch4_mt is None:
    initial_ch4_mt = (emission_co2_gt * 1000.0) / fair_utils.GWP20_CH4

  idx_emis = fair_utils.TARGET_YEAR - fair_utils.START_YEAR
  idx_sched_cdr = scheduled_cdr_year - fair_utils.START_YEAR
  idx_eval_start = (fair_utils.TARGET_YEAR + 1) - fair_utils.MIN_YEAR
  idx_eval_sched_end = scheduled_cdr_year - fair_utils.MIN_YEAR

  base_cfg = configs['baseline_v1.4.1_ch4mod']
  f_base_ref = run_calibration_sensitivity.run_fair_config(
      base_cfg, max_year=2055, progress=False
  )
  t_base_ref = f_base_ref.temperature.sel(
      scenario=fair_utils.SCENARIO, layer=0
  ).squeeze(drop=True)
  initial_ch4_p50_mt, _ = (
      run_calibration_sensitivity.calc_min_ch4_delay_for_config(
          cfg=base_cfg,
          delay_years=scheduled_cdr_year - fair_utils.TARGET_YEAR,
          t_base=t_base_ref,
          eval_percentile=0.50,
          max_year=2055,
      )
  )

  co2_no_cdr = np.zeros(1000)
  co2_no_cdr[idx_emis] = emission_co2_gt

  co2_sched_2042 = co2_no_cdr.copy()
  co2_sched_2042[idx_sched_cdr] = -emission_co2_gt

  ch4_p95_init = np.zeros(1000)
  ch4_p95_init[idx_emis] = -initial_ch4_mt

  ch4_p50_init = np.zeros(1000)
  ch4_p50_init[idx_emis] = -initial_ch4_p50_mt

  summary_rows = []
  traj_dfs = []

  for cfg_key, cfg in configs.items():
    print(
        f'\nEvaluating {reassessment_year} mid-course reassessment under:'
        f" {cfg['label']}..."
    )
    f_base = run_calibration_sensitivity.run_fair_config(
        cfg, max_year=sim_max_year, progress=True
    )
    t_base = f_base.temperature.sel(
        scenario=fair_utils.SCENARIO, layer=0
    ).squeeze(drop=True)

    f_nocdr = run_calibration_sensitivity.run_fair_config(
        cfg, co2_adj=co2_no_cdr, ch4_adj=ch4_p95_init, max_year=sim_max_year
    )
    t_nocdr = f_nocdr.temperature.sel(
        scenario=fair_utils.SCENARIO, layer=0
    ).squeeze(drop=True)
    df_nocdr = _extract_quantiles_df(t_nocdr, t_base, cfg_key, 'no_cdr')
    traj_dfs.append(df_nocdr)

    years_arr = df_nocdr['year'].values
    cont_cross_p95, first_pos_tb_p95 = _find_zero_crossing_year(
        years_arr, df_nocdr['p95'].values, after_year=fair_utils.TARGET_YEAR + 1
    )
    cont_cross_p50, _ = _find_zero_crossing_year(
        years_arr, df_nocdr['p50'].values, after_year=fair_utils.TARGET_YEAR + 1
    )

    latest_safe_cdr_year = (
        (first_pos_tb_p95 - 1) if first_pos_tb_p95 is not None else sim_max_year
    )
    cdr_advance_years = max(0, scheduled_cdr_year - latest_safe_cdr_year)

    f_unadj = run_calibration_sensitivity.run_fair_config(
        cfg, co2_adj=co2_sched_2042, ch4_adj=ch4_p95_init, max_year=sim_max_year
    )
    t_unadj = f_unadj.temperature.sel(
        scenario=fair_utils.SCENARIO, layer=0
    ).squeeze(drop=True)
    df_unadj = _extract_quantiles_df(
        t_unadj, t_base, cfg_key, 'unadjusted_2042_cdr'
    )
    traj_dfs.append(df_unadj)

    max_p95_bridge_unadj = float(
        df_unadj['p95'].values[idx_eval_start : idx_eval_sched_end + 1].max()
    )
    max_p50_bridge_unadj = float(
        df_unadj['p50'].values[idx_eval_start : idx_eval_sched_end + 1].max()
    )

    adv_cdr_year = min(
        scheduled_cdr_year, max(reassessment_year, latest_safe_cdr_year)
    )
    co2_adv = co2_no_cdr.copy()
    co2_adv[adv_cdr_year - fair_utils.START_YEAR] = -emission_co2_gt
    f_adv = run_calibration_sensitivity.run_fair_config(
        cfg, co2_adj=co2_adv, ch4_adj=ch4_p95_init, max_year=sim_max_year
    )
    t_adv = f_adv.temperature.sel(
        scenario=fair_utils.SCENARIO, layer=0
    ).squeeze(drop=True)
    traj_dfs.append(
        _extract_quantiles_df(
            t_adv, t_base, cfg_key, f'adapted_advance_cdr_{adv_cdr_year}'
        )
    )

    topup_p95_mt, topup_iters, f_topup = _solve_midcourse_ch4_topup(
        cfg=cfg,
        t_base=t_base,
        co2_sched=co2_sched_2042,
        base_ch4_sched=ch4_p95_init,
        reassessment_year=reassessment_year,
        scheduled_cdr_year=scheduled_cdr_year,
        eval_percentile=fair_utils.EVAL_PERCENTILE,
        tolerance=fair_utils.TOLERANCE,
        sim_max_year=sim_max_year,
    )
    t_topup = f_topup.temperature.sel(
        scenario=fair_utils.SCENARIO, layer=0
    ).squeeze(drop=True)
    traj_dfs.append(
        _extract_quantiles_df(
            t_topup, t_base, cfg_key, f'adapted_topup_{reassessment_year}_ch4'
        )
    )

    topup_from_p50_to_p50_mt, _, _ = _solve_midcourse_ch4_topup(
        cfg=cfg,
        t_base=t_base,
        co2_sched=co2_sched_2042,
        base_ch4_sched=ch4_p50_init,
        reassessment_year=reassessment_year,
        scheduled_cdr_year=scheduled_cdr_year,
        eval_percentile=0.50,
        tolerance=fair_utils.TOLERANCE,
        sim_max_year=sim_max_year,
    )

    summary_rows.append({
        'Config_Key': cfg_key,
        'Config_Label': cfg['label'],
        'Initial_2030_CH4_Mt': initial_ch4_mt,
        'Scheduled_CDR_Year': scheduled_cdr_year,
        'Reassessment_Year': reassessment_year,
        'Crossing_Year_Continuous_p95': cont_cross_p95,
        'Crossing_Year_Continuous_p50': cont_cross_p50,
        'Latest_Safe_CDR_Year_p95': latest_safe_cdr_year,
        'Advance_CDR_Years_Needed': cdr_advance_years,
        'Max_Unadjusted_Bridge_dT_p95_degC': max_p95_bridge_unadj,
        'Max_Unadjusted_Bridge_dT_p50_degC': max_p50_bridge_unadj,
        'Midcourse_2037_TopUp_CH4_Mt_p95': topup_p95_mt,
        'Midcourse_2037_TopUp_tCH4_per_tCO2': topup_p95_mt / 1000.0,
        'Midcourse_2037_TopUp_GWP20_tCO2e_per_tCO2': (
            (topup_p95_mt * fair_utils.GWP20_CH4) / 1000.0
        ),
        'Midcourse_2037_TopUp_GWP100_tCO2e_per_tCO2': (
            (topup_p95_mt * fair_utils.GWP100_CH4) / 1000.0
        ),
        'Counterfactual_p50Sized_TopUp_CH4_Mt_at_p50': topup_from_p50_to_p50_mt,
        'Solver_Iterations': topup_iters,
    })

  return pd.DataFrame(summary_rows), pd.concat(traj_dfs, ignore_index=True)


def plot_midcourse_figure(
    df_delay_traj,
    df_delay_summary,
    df_mid_traj,
    df_mid_summary,
    plot_year_min=2028,
    plot_year_max=2060,
    include_solver_delay_bridge=True,
    axis_fontsize=12,
    annot_fontsize=11,
    legend_fontsize=10.5,
    output_filename='Figure_4.png',
):
  """Generates Figure 4 demonstrating dynamic portfolio adaptation.

  Panel (a): 5-year durable CDR delivery delay (2042 to 2047) under the
    baseline configuration, comparing no adaptation (red) with model-optimized
    (green) and standardized Table S1 (blue) CH4 bridges.
  Panel (b): 2037 CH4 lifetime revision (-15% unperturbed CH4 lifetime),
    comparing no adaptation (red) with advancing CDR to 2040 (green) or adding
    a 2037 CH4 bridge (blue).

  Args:
    df_delay_traj: Trajectory quantile DataFrame from run_cdr_delay_simulation.
    df_delay_summary: Summary DataFrame from run_cdr_delay_simulation.
    df_mid_traj: Trajectory quantile DataFrame from run_midcourse_simulation.
    df_mid_summary: Summary DataFrame from run_midcourse_simulation.
    plot_year_min: Minimum year displayed on the x-axis.
    plot_year_max: Maximum year displayed on the x-axis.
    include_solver_delay_bridge: Whether to plot the model-optimized CH4 bridge
      in Panel (a) alongside the standardized Table S1 bridge.
    axis_fontsize: Font size for axis labels and tick labels.
    annot_fontsize: Font size for vertical-line text annotations.
    legend_fontsize: Font size for panel legends.
    output_filename: Filename saved inside fair_utils.TARGET_DIR.

  Returns:
    Full file path to the saved PNG figure.
  """
  _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)

  # ---------------------------------------------------------------------------
  # Panel (a): Durable CDR delivery delayed by 5 years (2042 -> 2047)
  # ---------------------------------------------------------------------------
  delay_row = df_delay_summary.iloc[0]
  delay_cfg_key = delay_row['Config_Key']
  sched_yr_a = int(round(delay_row['Scheduled_CDR_Year']))
  delayed_yr_a = int(round(delay_row['CDR_Delay_Years'] + sched_yr_a))
  std_bridge_mt = float(delay_row['Standardized_Bridge_CH4_Mt'])
  opt_bridge_mt = float(delay_row['Optimized_Bridge_CH4_Mt'])
  std_bridge_gwp20 = (std_bridge_mt * fair_utils.GWP20_CH4) / 1000.0
  opt_bridge_gwp20 = (opt_bridge_mt * fair_utils.GWP20_CH4) / 1000.0

  ax1.spines['top'].set_visible(False)
  ax1.spines['right'].set_visible(False)
  ax1.axhline(0, color='gray', linestyle='-', linewidth=1.2, zorder=1)

  delay_specs = [
      (
          'unadjusted_delayed_cdr',
          'No adaptation',
          '#E15759',
      ),
  ]
  if include_solver_delay_bridge:
    delay_specs.append((
        'adapted_opt_ch4_bridge',
        (
            f'Adapted (optimized):\n+{opt_bridge_gwp20:.2f} Gt CO₂e (GWP20)'
            f' CH₄\nin {sched_yr_a}'
        ),
        '#59A14F',
    ))
  delay_specs.append((
      'adapted_std_ch4_bridge',
      (
          f'Adapted (standardized):\n+{std_bridge_gwp20:.2f} Gt CO₂e (GWP20)'
          f' CH₄\nin {sched_yr_a}'
      ),
      '#4E79A7',
  ))

  ymax_a = 0.0
  for traj_name, label, col in delay_specs:
    sub = df_delay_traj[
        (df_delay_traj['config_key'] == delay_cfg_key)
        & (df_delay_traj['trajectory'] == traj_name)
        & (df_delay_traj['year'] >= plot_year_min)
        & (df_delay_traj['year'] <= plot_year_max)
    ].copy()
    for q_col in ('p05', 'p50', 'p95'):
      sub[q_col] *= 1000.0

    ax1.plot(
        sub['year'],
        sub['p50'],
        color=col,
        label=label,
        linewidth=2.5,
        zorder=3,
    )
    ax1.fill_between(
        sub['year'],
        sub['p05'],
        sub['p95'],
        color=col,
        alpha=0.15,
        zorder=2,
    )
    ymax_a = max(ymax_a, float(sub['p95'].max()))

  ymin_a_pad = -1.1
  ymax_a_pad = max(ymax_a * 1.45, 0.34)
  ax1.set_ylim(ymin_a_pad, ymax_a_pad)
  ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.4))

  ax1.axvline(sched_yr_a, color='gray', linestyle='--', linewidth=1.2, zorder=1)
  ax1.text(
      sched_yr_a - 0.6,
      ymax_a_pad * 0.54,
      f'{sched_yr_a} CDR delayed\nto {delayed_yr_a}',
      ha='right',
      va='center',
      fontsize=annot_fontsize,
      color='black',
  )

  ax1.text(
      -0.11,
      1.04,
      '(a)',
      transform=ax1.transAxes,
      fontsize=axis_fontsize,
      va='top',
  )
  ax1.set_ylabel(
      'Δ Temperature (10⁻³ °C)', fontsize=axis_fontsize, weight='medium'
  )
  ax1.tick_params(axis='both', which='major', labelsize=axis_fontsize)
  ax1.legend(frameon=False, loc='lower left', fontsize=legend_fontsize)

  # ---------------------------------------------------------------------------
  # Panel (b): 2037 CH4 lifetime revision (-15% unperturbed CH4 lifetime)
  # ---------------------------------------------------------------------------
  non_base = df_mid_summary[
      df_mid_summary['Config_Key'] != 'baseline_v1.4.1_ch4mod'
  ]
  focus_row = non_base.sort_values(
      'Midcourse_2037_TopUp_CH4_Mt_p95', ascending=False
  ).iloc[0]
  focus_key = focus_row['Config_Key']
  sched_yr_b = int(round(focus_row['Scheduled_CDR_Year']))
  reassess_yr = int(round(focus_row['Reassessment_Year']))
  adv_yr = int(
      min(
          sched_yr_b,
          max(reassess_yr, round(focus_row['Latest_Safe_CDR_Year_p95'])),
      )
  )
  topup_mt = float(focus_row['Midcourse_2037_TopUp_CH4_Mt_p95'])
  topup_gwp20 = (topup_mt * fair_utils.GWP20_CH4) / 1000.0

  ax2.spines['top'].set_visible(False)
  ax2.spines['right'].set_visible(False)
  ax2.axhline(0, color='gray', linestyle='-', linewidth=1.2, zorder=1)

  adapt_specs = [
      (
          'unadjusted_2042_cdr',
          'No adaptation',
          '#E15759',
      ),
      (
          f'adapted_advance_cdr_{adv_yr}',
          f'Adapted: advance CDR to {adv_yr}',
          '#59A14F',
      ),
      (
          f'adapted_topup_{reassess_yr}_ch4',
          f'Adapted: +{topup_gwp20:.2f} Gt CO₂e (GWP20) CH₄ in {reassess_yr}',
          '#4E79A7',
      ),
  ]

  ymax_b = 0.0
  for traj_name, label, col in adapt_specs:
    sub = df_mid_traj[
        (df_mid_traj['config_key'] == focus_key)
        & (df_mid_traj['trajectory'] == traj_name)
        & (df_mid_traj['year'] >= plot_year_min)
        & (df_mid_traj['year'] <= plot_year_max)
    ].copy()
    for q_col in ('p05', 'p50', 'p95'):
      sub[q_col] *= 1000.0

    ax2.plot(
        sub['year'],
        sub['p50'],
        color=col,
        label=label,
        linewidth=2.5,
        zorder=3,
    )
    ax2.fill_between(
        sub['year'],
        sub['p05'],
        sub['p95'],
        color=col,
        alpha=0.15,
        zorder=2,
    )
    ymax_b = max(ymax_b, float(sub['p95'].max()))

  ymin_b_pad = -0.6
  ymax_b_pad = max(ymax_b * 2.35, 0.17)
  ax2.set_ylim(ymin_b_pad, ymax_b_pad)
  ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

  ax2.axvline(
      reassess_yr, color='gray', linestyle='--', linewidth=1.2, zorder=1
  )
  ax2.text(
      reassess_yr - 0.6,
      ymax_b_pad * 0.54,
      f'CH₄ lifetime\nrevision in {reassess_yr}',
      ha='right',
      va='center',
      fontsize=annot_fontsize,
      color='black',
  )

  ax2.text(
      -0.11,
      1.04,
      '(b)',
      transform=ax2.transAxes,
      fontsize=axis_fontsize,
      va='top',
  )
  ax2.set_ylabel(
      'Δ Temperature (10⁻³ °C)', fontsize=axis_fontsize, weight='medium'
  )
  ax2.set_xlabel('Year', fontsize=axis_fontsize, weight='medium')
  ax2.tick_params(axis='both', which='major', labelsize=axis_fontsize)
  ax2.set_xlim(plot_year_min, plot_year_max)
  ax2.set_xticks(np.arange(2030, plot_year_max + 1, 5))
  ax2.legend(frameon=False, loc='lower right', fontsize=legend_fontsize)

  plt.tight_layout()

  os.makedirs(fair_utils.TARGET_DIR, exist_ok=True)
  fig_path = os.path.join(fair_utils.TARGET_DIR, output_filename)
  plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
  print(f'\nFigure 4 saved to: {fig_path}')

  plt.show()
  return fig_path


def execute_midcourse_reassessment(
    ch4_perturbation_pct=-15.0,
    scheduled_cdr_year=2042,
    cdr_delay_years=5,
    reassessment_year=2037,
    recalc_table_s5=False,
    recalc_midcourse=False,
    recalc_delay=False,
    skip_calc=False,
    make_plot=True,
    include_solver_delay_bridge=True,
    output_filename='Figure_4.png',
):
  """Executes Table S5 sensitivity, portfolio adaptation runs, and Figure 4.

  Automatically reuses pre-existing stage CSVs in fair_utils.TARGET_DIR unless
  the corresponding recalc_* flag is True.

  Args:
    ch4_perturbation_pct: Percentage change in unperturbed CH4 lifetime for the
      hypothetical sensitivity configuration.
    scheduled_cdr_year: Originally contracted durable CDR delivery year.
    cdr_delay_years: Length of durable CDR delivery delay in Panel (a).
    reassessment_year: Year of CH4 lifetime reassessment in Panel (b).
    recalc_table_s5: Force recalculation of Supplementary Table S5 CSVs.
    recalc_midcourse: Force recalculation of 2037 reassessment CSVs.
    recalc_delay: Force recalculation of 5-year CDR delay CSVs.
    skip_calc: Reload all existing stage CSVs without re-running FaIR.
    make_plot: Whether to render and save Figure 4.
    include_solver_delay_bridge: Whether to plot the model-optimized CH4 bridge
      in Figure 4a alongside the standardized Table S1 bridge.
    output_filename: Filename for Figure 4 inside fair_utils.TARGET_DIR.

  Returns:
    Tuple of (df_s5_raw, df_s5_fmt, df_delay_summary, df_mid_summary).
  """
  os.makedirs(fair_utils.TARGET_DIR, exist_ok=True)
  mid_summary_csv = os.path.join(
      fair_utils.TARGET_DIR, 'midcourse_reassessment_summary.csv'
  )
  mid_traj_csv = os.path.join(
      fair_utils.TARGET_DIR, 'midcourse_temperature_trajectories.csv'
  )
  delay_summary_csv = os.path.join(
      fair_utils.TARGET_DIR, 'cdr_delay_summary.csv'
  )
  delay_traj_csv = os.path.join(
      fair_utils.TARGET_DIR, 'cdr_delay_temperature_trajectories.csv'
  )

  # 1. Supplementary Table S5 stage
  df_s5_raw, df_s5_fmt = (
      run_calibration_sensitivity.execute_calibration_sensitivity(
          ch4_perturbation_pct=ch4_perturbation_pct,
          skip_calc=(skip_calc or not recalc_table_s5),
          print_summary=False,
      )
  )

  configs = None

  # 2. 2037 CH4 lifetime reassessment stage (Figure 4b)
  if (
      (skip_calc or not recalc_midcourse)
      and os.path.exists(mid_summary_csv)
      and os.path.exists(mid_traj_csv)
  ):
    print('Loading saved 2037 reassessment CSVs...')
    df_mid_summary = pd.read_csv(mid_summary_csv)
    df_mid_traj = pd.read_csv(mid_traj_csv)
  else:
    configs = run_calibration_sensitivity.ensure_calibration_files(
        data_dir=fair_utils.DATA_DIR, ch4_perturbation_pct=ch4_perturbation_pct
    )
    df_mid_summary, df_mid_traj = run_midcourse_simulation(
        configs=configs,
        scheduled_cdr_year=scheduled_cdr_year,
        reassessment_year=reassessment_year,
    )
    df_mid_summary.to_csv(mid_summary_csv, index=False)
    df_mid_traj.to_csv(mid_traj_csv, index=False)

  # 3. 5-year durable CDR delivery delay stage (Figure 4a)
  if (
      (skip_calc or not recalc_delay)
      and os.path.exists(delay_summary_csv)
      and os.path.exists(delay_traj_csv)
  ):
    print('Loading saved 5-year CDR delivery delay CSVs...')
    df_delay_summary = pd.read_csv(delay_summary_csv)
    df_delay_traj = pd.read_csv(delay_traj_csv)
  else:
    if configs is None:
      configs = run_calibration_sensitivity.ensure_calibration_files(
          data_dir=fair_utils.DATA_DIR,
          ch4_perturbation_pct=ch4_perturbation_pct,
      )
    df_delay_summary, df_delay_traj = run_cdr_delay_simulation(
        configs=configs,
        df_s5_raw=df_s5_raw,
        scheduled_cdr_year=scheduled_cdr_year,
        cdr_delay_years=cdr_delay_years,
    )
    df_delay_summary.to_csv(delay_summary_csv, index=False)
    df_delay_traj.to_csv(delay_traj_csv, index=False)
    print(f'Saved CDR delay summary to: {delay_summary_csv}')

  run_calibration_sensitivity.print_sensitivity_summary(
      df_s5_raw, df_s5_fmt, ch4_perturbation_pct
  )

  if make_plot:
    plot_midcourse_figure(
        df_delay_traj=df_delay_traj,
        df_delay_summary=df_delay_summary,
        df_mid_traj=df_mid_traj,
        df_mid_summary=df_mid_summary,
        include_solver_delay_bridge=include_solver_delay_bridge,
        output_filename=output_filename,
    )

  return df_s5_raw, df_s5_fmt, df_delay_summary, df_mid_summary


# Backward-compatible alias for existing notebook calls
execute_all_s5_analyses = execute_midcourse_reassessment


if __name__ == '__main__':
  execute_midcourse_reassessment(skip_calc=False)
