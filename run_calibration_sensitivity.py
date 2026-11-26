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

"""Evaluates sensitivity of standardized CH4 ratios to FaIR calibrations (Table S5)."""

import inspect
import os
import time

from fair import FAIR
from fair import interface
from fair.io import read_properties
import fair_utils
import numpy as np
import pandas as pd
import pooch
import xarray as xr

# Zenodo DOIs for published fair-calibrate parameter releases
ZENODO_V141_DOI = 'doi:10.5281/zenodo.10566813'
ZENODO_V160_DOI = 'doi:10.5281/zenodo.18828694'


def _retrieve_doi_with_retry(doi_url, fname, path, max_retries=4, timeout=120):
  """Downloads a calibration file from Zenodo via pooch with retry backoff."""
  downloader = pooch.DOIDownloader(
      timeout=timeout,
      headers={
          'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) FaIR-Calibration/1.0'
      },
  )
  for attempt in range(1, max_retries + 1):
    try:
      return pooch.retrieve(
          url=doi_url,
          known_hash=None,
          fname=fname,
          path=path,
          downloader=downloader,
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      if attempt == max_retries:
        raise
      wait_s = 5 * attempt
      print(
          f'  Zenodo timeout/error (attempt {attempt}/{max_retries}): {exc}\n'
          f'  Retrying in {wait_s}s...'
      )
      time.sleep(wait_s)


def ensure_calibration_files(
    data_dir=fair_utils.DATA_DIR, ch4_perturbation_pct=-15.0
):
  """Ensures calibration CSVs exist in data_dir and returns configuration specs.

  Args:
    data_dir: Directory containing FaIR input and calibration CSV files.
    ch4_perturbation_pct: Percentage change applied to unperturbed CH4 lifetime
      in the hypothetical sensitivity configuration (e.g., -15.0 for -15%).

  Returns:
    Dictionary mapping configuration keys to parameter dictionaries.
  """
  os.makedirs(data_dir, exist_ok=True)

  params_160 = os.path.join(
      data_dir, 'calibrated_constrained_parameters_calibration1.6.0.csv'
  )
  if not os.path.exists(params_160):
    print('Downloading v1.6.0 calibrated_constrained_parameters.csv...')
    _retrieve_doi_with_retry(
        doi_url=f'{ZENODO_V160_DOI}/calibrated_constrained_parameters.csv',
        fname='calibrated_constrained_parameters_calibration1.6.0.csv',
        path=data_dir,
    )

  species_160 = os.path.join(
      data_dir, 'species_configs_properties_calibration1.6.0.csv'
  )
  if not os.path.exists(species_160):
    print('Downloading v1.6.0 species_configs_properties.csv...')
    _retrieve_doi_with_retry(
        doi_url=f'{ZENODO_V160_DOI}/species_configs_properties.csv',
        fname='species_configs_properties_calibration1.6.0.csv',
        path=data_dir,
    )

  species_141_unmod = os.path.join(
      data_dir, 'species_configs_properties_calibration1.4.1.csv'
  )
  if not os.path.exists(species_141_unmod):
    try:
      print('Downloading unmodified v1.4.1 species_configs_properties.csv...')
      _retrieve_doi_with_retry(
          doi_url=f'{ZENODO_V141_DOI}/species_configs_properties.csv',
          fname='species_configs_properties_calibration1.4.1.csv',
          path=data_dir,
          max_retries=2,
          timeout=60,
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      print(
          '  Falling back to local reconstruction for unmodified v1.4.1'
          f' ({exc})...'
      )
      df_mod = pd.read_csv(fair_utils.SPECIES_FILE, index_col=0)
      df_160 = pd.read_csv(species_160, index_col=0)
      if 'CH4' in df_mod.index and 'CH4' in df_160.index:
        shared_ch4_cols = [
            c
            for c in df_mod.columns
            if c in df_160.columns and ('lifetime' in c or 'ch4' in c)
        ]
        df_mod.loc['CH4', shared_ch4_cols] = df_160.loc['CH4', shared_ch4_cols]
      df_mod.to_csv(species_141_unmod)

  ch4_scale = 1.0 + (ch4_perturbation_pct / 100.0)
  if ch4_perturbation_pct < 0:
    pert_label = (
        f'{abs(ch4_perturbation_pct):.0f}% shorter unperturbed CH4 lifetime'
    )
  else:
    pert_label = f'{ch4_perturbation_pct:+.0f}% unperturbed CH4 lifetime'

  return {
      'baseline_v1.4.1_ch4mod': {
          'key': 'baseline_v1.4.1_ch4mod',
          'label': 'Baseline v1.4.1 + 1.4.0 CH4',
          'params_file': fair_utils.PARAMS_FILE,
          'species_file': fair_utils.SPECIES_FILE,
          'use_thornhill': False,
          'ch4_lifetime_scale': 1.0,
          'ch4_forcing_scale': 1.0,
      },
      'unmodified_v1.4.1': {
          'key': 'unmodified_v1.4.1',
          'label': 'Unmodified v1.4.1',
          'params_file': fair_utils.PARAMS_FILE,
          'species_file': species_141_unmod,
          'use_thornhill': True,
          'ch4_lifetime_scale': 1.0,
          'ch4_forcing_scale': 1.0,
      },
      'v1.6.0_fastmip': {
          'key': 'v1.6.0_fastmip',
          'label': 'CMIP7/IGCC v1.6.0 (fastmip v1)',
          'params_file': params_160,
          'species_file': species_160,
          'use_thornhill': True,
          'ch4_lifetime_scale': 1.0,
          'ch4_forcing_scale': 1.0,
      },
      'perturbed_ch4': {
          'key': 'perturbed_ch4',
          'label': pert_label,
          'params_file': fair_utils.PARAMS_FILE,
          'species_file': fair_utils.SPECIES_FILE,
          'use_thornhill': False,
          'ch4_lifetime_scale': ch4_scale,
          'ch4_forcing_scale': 1.0,
      },
  }


def _ensure_compatible_driving_csvs(
    cfg, data_dir=fair_utils.DATA_DIR, scenario=fair_utils.SCENARIO
):
  """Ensures all species in cfg['species_file'] exist in driving CSVs.

  Patches zero-baseline rows (such as 'Irrigation' in fair-calibrate v1.6.0)
  into cached compatibility CSVs when absent from baseline driving files.

  Args:
    cfg: Configuration dictionary from ensure_calibration_files.
    data_dir: Directory containing driving emissions and forcing CSVs.
    scenario: Scenario name in the driving CSV files.

  Returns:
    Tuple of (compatible_emissions_csv_path, compatible_forcing_csv_path).
  """
  _, properties = read_properties(filename=cfg['species_file'])
  emis_path = os.path.join(data_dir, fair_utils.EMISSIONS_FILE)
  forc_path = os.path.join(data_dir, fair_utils.FORCING_FILE)

  def _patch_csv_if_needed(src_path, needed_species, default_unit, tag):
    df = pd.read_csv(src_path)
    col_map = {c.lower(): c for c in df.columns}
    scen_col = col_map['scenario']
    var_col = col_map['variable']
    unit_col = col_map.get('unit')

    df_scen = df[df[scen_col] == scenario]
    present = set(df_scen[var_col].unique())
    missing = [s for s in needed_species if s not in present]
    if not missing:
      return src_path

    compat_path = os.path.join(
        data_dir, f"_compat_{tag}_{os.path.basename(cfg['species_file'])}"
    )
    if os.path.exists(compat_path):
      df_cached = pd.read_csv(compat_path)
      cached_present = set(
          df_cached[df_cached[scen_col] == scenario][var_col].unique()
      )
      if all(s in cached_present for s in needed_species):
        return compat_path

    year_cols = []
    for col in df.columns:
      if col in (scen_col, var_col, unit_col):
        continue
      try:
        float(col)
        year_cols.append(col)
      except ValueError:
        pass

    template_row = df_scen.iloc[0].copy()
    extra_rows = []
    for sp in missing:
      new_row = template_row.copy()
      new_row[var_col] = sp
      if unit_col is not None:
        new_row[unit_col] = default_unit
      for yc in year_cols:
        new_row[yc] = 0.0
      extra_rows.append(new_row)

    df_out = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)
    df_out.to_csv(compat_path, index=False)
    return compat_path

  needed_emis = [
      s for s, p in properties.items() if p['input_mode'] == 'emissions'
  ]
  needed_forc = [
      s for s, p in properties.items() if p['input_mode'] == 'forcing'
  ]

  compat_emis = _patch_csv_if_needed(
      emis_path, needed_emis, 'Mt yr^-1', 'emissions'
  )
  compat_forc = _patch_csv_if_needed(
      forc_path, needed_forc, 'W m^-2', 'forcing'
  )
  return compat_emis, compat_forc


def run_fair_config(
    cfg,
    co2_adj=None,
    ch4_adj=None,
    hfc_adj=None,
    hfc_specie=fair_utils.HFC_SPECIE,
    scenario=fair_utils.SCENARIO,
    min_year=fair_utils.MIN_YEAR,
    max_year=fair_utils.MAX_YEAR,
    progress=False,
):
  """Runs FaIR under a specified calibration configuration and perturbation.

  Args:
    cfg: Configuration dictionary from ensure_calibration_files.
    co2_adj: Array of CO2 emissions adjustments (Gt CO2/yr from START_YEAR).
    ch4_adj: Array of CH4 emissions adjustments (Mt CH4/yr from START_YEAR).
    hfc_adj: Array of HFC emissions adjustments (kt HFC/yr from START_YEAR).
    hfc_specie: Name of the HFC species to adjust.
    scenario: Baseline scenario identifier.
    min_year: Start year of the FaIR simulation.
    max_year: End year of the FaIR simulation.
    progress: Whether to display FaIR's progress bar.

  Returns:
    Configured and executed FAIR model instance.
  """
  if co2_adj is None:
    co2_adj = np.zeros(1000)
  if ch4_adj is None:
    ch4_adj = np.zeros(1000)
  if hfc_adj is None:
    hfc_adj = np.zeros(1000)

  compat_emis_file, compat_forc_file = _ensure_compatible_driving_csvs(
      cfg, data_dir=fair_utils.DATA_DIR, scenario=scenario
  )

  if cfg.get('use_thornhill', False):
    f = FAIR(ch4_method='Thornhill2021')
  else:
    f = FAIR()

  f.define_time(min_year, max_year, 1)
  f.define_scenarios([scenario])

  df_configs = pd.read_csv(cfg['params_file'], index_col=0)
  f.define_configs(df_configs.index)
  species, properties = read_properties(filename=cfg['species_file'])
  f.define_species(species, properties)
  f.allocate()

  f.fill_from_csv(
      emissions_file=compat_emis_file,
      forcing_file=compat_forc_file,
  )

  time_mask = f.emissions['timepoints'] >= fair_utils.START_YEAR
  time_sel = f.emissions['timepoints'].where(time_mask, drop=True)
  n_years = time_sel.size

  if np.any(co2_adj[:n_years] != 0):
    adj_da_co2 = xr.DataArray(
        co2_adj[:n_years], coords={'timepoints': time_sel}, dims='timepoints'
    )
    co2_slice = dict(timepoints=time_sel, specie='CO2 FFI')
    f.emissions.loc[co2_slice] = f.emissions.sel(**co2_slice) + adj_da_co2

  if np.any(ch4_adj[:n_years] != 0):
    adj_da_ch4 = xr.DataArray(
        ch4_adj[:n_years], coords={'timepoints': time_sel}, dims='timepoints'
    )
    ch4_slice = dict(timepoints=time_sel, specie='CH4')
    f.emissions.loc[ch4_slice] = f.emissions.sel(**ch4_slice) + adj_da_ch4

  if np.any(hfc_adj[:n_years] != 0):
    adj_da_hfc = xr.DataArray(
        hfc_adj[:n_years], coords={'timepoints': time_sel}, dims='timepoints'
    )
    hfc_slice = dict(timepoints=time_sel, specie=hfc_specie)
    f.emissions.loc[hfc_slice] = f.emissions.sel(**hfc_slice) + adj_da_hfc

  for exog_specie in ['Volcanic', 'Solar', 'Irrigation']:
    col_name = f'forcing_scale[{exog_specie}]'
    if exog_specie in species and col_name in df_configs.columns:
      interface.fill(
          f.forcing,
          f.forcing.sel(specie=exog_specie)
          * df_configs[col_name].values.squeeze(),
          specie=exog_specie,
      )

  f.fill_species_configs(cfg['species_file'])
  f.override_defaults(cfg['params_file'])

  if cfg.get('ch4_lifetime_scale', 1.0) != 1.0:
    f.species_configs['unperturbed_lifetime'].loc[dict(specie='CH4')] = (
        f.species_configs['unperturbed_lifetime'].sel(specie='CH4')
        * cfg['ch4_lifetime_scale']
    )
  if cfg.get('ch4_forcing_scale', 1.0) != 1.0:
    f.species_configs['forcing_scale'].loc[dict(specie='CH4')] = (
        f.species_configs['forcing_scale'].sel(specie='CH4')
        * cfg['ch4_forcing_scale']
    )

  interface.initialise(
      f.concentration, f.species_configs['baseline_concentration']
  )
  interface.initialise(f.forcing, 0)
  interface.initialise(f.temperature, 0)
  interface.initialise(f.cumulative_emissions, 0)
  interface.initialise(f.airborne_emissions, 0)
  interface.initialise(f.ocean_heat_content_change, 0)

  if 'progress' in inspect.signature(f.run).parameters:
    f.run(progress=progress)
  else:
    f.run()

  return f


def calc_min_ch4_delay_for_config(
    cfg,
    delay_years,
    t_base,
    eval_percentile=fair_utils.EVAL_PERCENTILE,
    search_low=0.0,
    search_high=40.0,
    tolerance=fair_utils.TOLERANCE,
    max_year=fair_utils.MAX_YEAR,
):
  """Calculates minimum CH4 mitigation for a delayed CDR pulse under cfg.

  Uses binary search to identify the minimum CH4 mitigation pulse in TARGET_YEAR
  (in Mt CH4) that holds delta_T at eval_percentile <= 0 C from TARGET_YEAR + 1
  through TARGET_YEAR + delay_years for a 1 Gt CO2 pulse in TARGET_YEAR.

  Args:
    cfg: Configuration dictionary from ensure_calibration_files.
    delay_years: Number of years between CO2 emission and durable CDR pulse.
    t_base: Baseline surface temperature DataArray for cfg.
    eval_percentile: Ensemble quantile evaluated for warming neutralization.
    search_low: Lower bound for binary search (Mt CH4).
    search_high: Upper bound for binary search (Mt CH4).
    tolerance: Convergence tolerance (Mt CH4).
    max_year: End year of the FaIR simulation.

  Returns:
    Tuple of (required_ch4_mt, iterations).
  """
  idx_start_eval = (fair_utils.TARGET_YEAR + 1) - fair_utils.MIN_YEAR
  idx_end_eval = (fair_utils.TARGET_YEAR + delay_years) - fair_utils.MIN_YEAR

  low, high = search_low, search_high
  iteration = 0

  co2_pulse = np.zeros(1000)
  co2_pulse[fair_utils.OFFSET] = 1.0
  co2_pulse[fair_utils.OFFSET + delay_years] = -1.0

  while (high - low) > tolerance:
    iteration += 1
    mid_val = (low + high) / 2.0

    mitigation_pulse = np.zeros(1000)
    mitigation_pulse[fair_utils.OFFSET] = -mid_val

    f_scen = run_fair_config(
        cfg,
        co2_adj=co2_pulse,
        ch4_adj=mitigation_pulse,
        max_year=max_year,
        progress=False,
    )
    t_scen = f_scen.temperature.sel(
        scenario=fair_utils.SCENARIO, layer=0
    ).squeeze(drop=True)
    temp_diff = (t_scen - t_base).quantile(eval_percentile, dim='config')
    max_t_anomaly = float(
        temp_diff[idx_start_eval : idx_end_eval + 1].max().values
    )

    if max_t_anomaly > 0:
      low = mid_val
    else:
      high = mid_val

  return high, iteration


def print_sensitivity_summary(df_raw, df_formatted, ch4_perturbation_pct=-15.0):
  """Prints Supplementary Table S5 and key calibration sensitivity metrics."""
  print('\n' + '=' * 80)
  print('SUPPLEMENTARY TABLE S5 (Ton CH4 mitigation per ton CO2 emitted)')
  print('=' * 80)
  print(df_formatted.to_string(index=False))

  calib_keys = ['baseline_v1.4.1_ch4mod', 'unmodified_v1.4.1', 'v1.6.0_fastmip']
  df_p95 = df_raw[
      (df_raw['Percentile'] == 95) & (df_raw['Config_Key'].isin(calib_keys))
  ]

  piv_t = df_p95.pivot(
      index='Delay_Years', columns='Config_Key', values='Ton_CH4_per_Ton_CO2'
  )
  base_col = piv_t['baseline_v1.4.1_ch4mod']
  same_scheme_diff = (piv_t['v1.6.0_fastmip'] - piv_t['unmodified_v1.4.1']).abs()

  abs_diffs = pd.concat(
      [
          (piv_t['unmodified_v1.4.1'] - base_col).abs(),
          (piv_t['v1.6.0_fastmip'] - base_col).abs(),
          same_scheme_diff,
      ],
      axis=1,
  )
  pct_diffs = pd.concat(
      [
          ((piv_t['unmodified_v1.4.1'] - base_col).abs() / base_col) * 100.0,
          ((piv_t['v1.6.0_fastmip'] - base_col).abs() / base_col) * 100.0,
          (same_scheme_diff / piv_t['unmodified_v1.4.1']) * 100.0,
      ],
      axis=1,
  )

  df_pert = df_raw[
      (df_raw['Percentile'] == 95) & (df_raw['Config_Key'] == 'perturbed_ch4')
  ].set_index('Delay_Years')['Ton_CH4_per_Ton_CO2']

  print('\n' + '=' * 80)
  print('CALIBRATION SENSITIVITY SUMMARY (p95)')
  print('=' * 80)
  print(
      '1. Unmodified v1.4.1 vs. CMIP7/IGCC v1.6.0 (same CH4 lifetime scheme):\n'
      f'   Max shift = {float(same_scheme_diff.max()):.4f} ton CH4 per ton CO2'
  )
  print(
      '2. Range across all three fair-calibrate configurations:\n'
      f'   {float(abs_diffs.min().min()):.3f} to'
      f' {float(abs_diffs.max().max()):.3f} ton CH4 per ton CO2'
      f' ({float(pct_diffs.min().min()):.1f}% to'
      f' {float(pct_diffs.max().max()):.1f}%)'
  )
  print(
      f'3. {abs(ch4_perturbation_pct):.0f}% shorter unperturbed CH4 lifetime:\n'
      f'   p95 ratios span {float(df_pert.min()):.3f} to'
      f' {float(df_pert.max()):.3f} ton CH4 per ton CO2'
      f' (vs. {float(base_col.min()):.3f} to {float(base_col.max()):.3f} in'
      ' baseline)'
  )
  print('=' * 80 + '\n')


def execute_calibration_sensitivity(
    delay_years_list=None,
    percentiles=None,
    ch4_perturbation_pct=-15.0,
    use_fast_horizon=True,
    skip_calc=False,
    print_summary=True,
):
  """Computes or loads Supplementary Table S5 calibration sensitivity results.

  Args:
    delay_years_list: List of durable CDR replacement horizons in years.
    percentiles: List of FaIR ensemble quantiles to evaluate.
    ch4_perturbation_pct: Percentage change in unperturbed CH4 lifetime for the
      hypothetical sensitivity configuration.
    use_fast_horizon: If True, truncates simulation horizon shortly after the
      maximum replacement year for faster execution.
    skip_calc: If True and saved CSV outputs exist, reloads them from disk.
    print_summary: Whether to print the formatted table and summary statistics.

  Returns:
    Tuple of (df_raw, df_formatted).
  """
  if delay_years_list is None:
    delay_years_list = [5, 10, 15, 20]
  if percentiles is None:
    percentiles = [0.50, 0.95]

  os.makedirs(fair_utils.TARGET_DIR, exist_ok=True)
  raw_csv_path = os.path.join(
      fair_utils.TARGET_DIR, 'table_s5_calibration_sensitivity_raw.csv'
  )
  fmt_csv_path = os.path.join(
      fair_utils.TARGET_DIR, 'table_s5_calibration_sensitivity_formatted.csv'
  )

  if skip_calc and os.path.exists(raw_csv_path) and os.path.exists(fmt_csv_path):
    print('Loading saved Supplementary Table S5 CSVs...')
    df_raw = pd.read_csv(raw_csv_path)
    df_formatted = pd.read_csv(fmt_csv_path)
    if print_summary:
      print_sensitivity_summary(df_raw, df_formatted, ch4_perturbation_pct)
    return df_raw, df_formatted

  configs = ensure_calibration_files(
      data_dir=fair_utils.DATA_DIR, ch4_perturbation_pct=ch4_perturbation_pct
  )

  sim_max_year = (
      (fair_utils.TARGET_YEAR + max(delay_years_list) + 5)
      if use_fast_horizon
      else fair_utils.MAX_YEAR
  )

  raw_rows = []
  for cfg_key, cfg in configs.items():
    print(f"\nRunning baseline for configuration: {cfg['label']}...")
    f_base = run_fair_config(cfg, max_year=sim_max_year, progress=True)
    t_base = f_base.temperature.sel(
        scenario=fair_utils.SCENARIO, layer=0
    ).squeeze(drop=True)

    for delay_yr in delay_years_list:
      for pct in percentiles:
        pct_int = int(round(pct * 100))
        req_ch4_mt, n_iter = calc_min_ch4_delay_for_config(
            cfg=cfg,
            delay_years=delay_yr,
            t_base=t_base,
            eval_percentile=pct,
            search_low=0.0,
            search_high=40.0,
            tolerance=fair_utils.TOLERANCE,
            max_year=sim_max_year,
        )
        ton_per_ton = req_ch4_mt / 1000.0
        gwp20_ratio = (req_ch4_mt * fair_utils.GWP20_CH4) / 1000.0
        gwp100_ratio = (req_ch4_mt * fair_utils.GWP100_CH4) / 1000.0

        raw_rows.append({
            'Config_Key': cfg_key,
            'Config_Label': cfg['label'],
            'Delay_Years': delay_yr,
            'Replacement_Year': fair_utils.TARGET_YEAR + delay_yr,
            'Percentile': pct_int,
            'Required_CH4_Mt_per_Gt_CO2': req_ch4_mt,
            'Ton_CH4_per_Ton_CO2': ton_per_ton,
            'Ton_CH4_CO2e_GWP20_per_Ton_CO2': gwp20_ratio,
            'Ton_CH4_CO2e_GWP100_per_Ton_CO2': gwp100_ratio,
            'Iterations': n_iter,
        })
        print(
            f'  Delay {delay_yr:2d} yr (p{pct_int:02d}):'
            f' {ton_per_ton:.3f} t CH4/t CO2 ({req_ch4_mt:.2f} Mt;'
            f' {n_iter} iters)'
        )

  df_raw = pd.DataFrame(raw_rows)
  df_raw.to_csv(raw_csv_path, index=False)
  print(f'\nRaw sensitivity data saved to: {raw_csv_path}')

  pert_col_header = (
      f'{abs(ch4_perturbation_pct):.0f}% shorter unperturbed CH4 lifetime'
      ' (ton CH4 mitigation per ton CO2 emitted, p95)'
  )

  fmt_rows = []
  for delay_yr in delay_years_list:
    sub = df_raw[df_raw['Delay_Years'] == delay_yr]

    def fmt_pair(key, df_sub=sub):
      p50_val = df_sub[
          (df_sub['Config_Key'] == key) & (df_sub['Percentile'] == 50)
      ]['Ton_CH4_per_Ton_CO2'].values[0]
      p95_val = df_sub[
          (df_sub['Config_Key'] == key) & (df_sub['Percentile'] == 95)
      ]['Ton_CH4_per_Ton_CO2'].values[0]
      return f'{p50_val:.3f} / {p95_val:.3f}'

    pert_p95 = sub[
        (sub['Config_Key'] == 'perturbed_ch4') & (sub['Percentile'] == 95)
    ]['Ton_CH4_per_Ton_CO2'].values[0]

    fmt_rows.append({
        'Latest year of replacement with durable CDR': delay_yr,
        'Baseline v1.4.1 + 1.4.0 CH4 (ton CH4 mitigation per ton CO2 emitted, p50 / p95)': (
            fmt_pair('baseline_v1.4.1_ch4mod')
        ),
        'Unmodified v1.4.1 (ton CH4 mitigation per ton CO2 emitted, p50 / p95)': (
            fmt_pair('unmodified_v1.4.1')
        ),
        'CMIP7/IGCC v1.6.0 (ton CH4 mitigation per ton CO2 emitted, p50 / p95)': (
            fmt_pair('v1.6.0_fastmip')
        ),
        pert_col_header: f'{pert_p95:.3f}',
    })

  df_formatted = pd.DataFrame(fmt_rows)
  df_formatted.to_csv(fmt_csv_path, index=False)
  print(f'Formatted Table S5 saved to: {fmt_csv_path}')

  if print_summary:
    print_sensitivity_summary(df_raw, df_formatted, ch4_perturbation_pct)

  return df_raw, df_formatted


if __name__ == '__main__':
  execute_calibration_sensitivity(skip_calc=False)
