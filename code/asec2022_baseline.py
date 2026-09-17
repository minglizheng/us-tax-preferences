"""


@author: mlzheng

Baseline estimation for tax year 2021
Married Filing Jointly, EITC and ACTC included as endogenous transfers
Data: CPS ASEC 2022 (pppub22.csv)

Two specifications:
1. Without EIP: EIP excluded from disposable income
2. With EIP: EIP included in disposable income

Reports three WAAD criteria:
1. Population-weighted
2. Tax-paid-weighted
3. Revenue-weighted

Also prints weighted ratios:
- Total EIP / Total AGI
- Total (EITC + ACTC) / Total AGI
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore', message='delta_grad == 0.0')

# =============================================================================
# SETTINGS
# =============================================================================

DATA_FILE = "pppub22.csv"           # 2022 ASEC, tax year 2021
FILING_STATUS_MAX = 4

STD_DEDUCTION = 25100
BRACKETS = np.array([19900, 81050, 172750, 329850, 418850, 628300])

# Preferential income calibration
LTCG_SHARE = 1.0
QDIV_SHARE = 0.75

# Use survey weights?
USE_WEIGHTS = True

# =============================================================================
# SEARCH GRID
# =============================================================================

# Wider grid to accommodate EIP specifications
g_fine = np.round(np.arange(0.100, 0.170, 0.001), 3)
g_report = np.array([0.110, 0.120, 0.130, 0.140, 0.150])

G_GRID = np.union1d(g_fine, g_report)
SIGMA_GRID_01 = np.round(np.arange(1.050, 1.260, 0.001), 3)

# Solver settings
SOLVER_OPTIONS = {
    'gtol': 1e-8,
    'xtol': 1e-8,
    'barrier_tol': 1e-8,
    'maxiter': 1000,
    'verbose': 0
}

# =============================================================================
# SPECIFICATIONS
# =============================================================================

SPECIFICATIONS = [
    {'label': 'Without EIP', 'include_eip': False},
    {'label': 'With EIP',    'include_eip': True},
]

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

print(f"Reading {DATA_FILE}...")

required_cols = [
    'FILESTAT', 'TAX_ID', 'AGI', 'CAP_VAL', 'DIV_VAL', 'TAX_INC',
    'FEDTAX_AC', 'FEDTAX_BC', 'SPM_EITC', 'SPM_ACTC',
    'SPM_CAPHOUSESUB', 'SPM_SNAPSUB', 'MARSUPWT',
    'EIP_CRD'
]

df_full = pd.read_csv(DATA_FILE)
missing_cols = [col for col in required_cols if col not in df_full.columns]
if missing_cols:
    print(f"Warning: Missing columns: {missing_cols}")

available_cols = [col for col in required_cols if col in df_full.columns]
data = df_full[available_cols].copy()

# EIP variable
if 'EIP_CRD' in data.columns:
    data['EIP'] = data['EIP_CRD'].fillna(0)
    print("Using EIP variable: EIP_CRD")
else:
    print("Warning: EIP_CRD not found. EIP set to zero.")
    data['EIP'] = 0

# Select married filing jointly
df0 = data[data['FILESTAT'] < FILING_STATUS_MAX].copy()

# Identify primary filer and aggregate spouse information
df0 = df0.sort_values(['TAX_ID', 'AGI'], ascending=[True, False])
df0['is_filer'] = df0.groupby('TAX_ID')['AGI'].transform(
    lambda x: (x == x.max()) & (x >= 0)
)
df0['total_cap'] = df0.groupby('TAX_ID')['CAP_VAL'].transform('sum')
df0['total_div'] = df0.groupby('TAX_ID')['DIV_VAL'].transform('sum')
df0['total_eip'] = df0.groupby('TAX_ID')['EIP'].transform('sum')

df0['filer_cap'] = df0['total_cap'] * df0['is_filer']
df0['filer_div'] = df0['total_div'] * df0['is_filer']
df0['filer_eip'] = df0['total_eip'] * df0['is_filer']

filer_data = df0[df0['is_filer']].copy()
filer_data = filer_data[filer_data['AGI'] > 0].copy()
df = filer_data

# =============================================================================
# SAMPLE DESCRIPTION
# =============================================================================

n_total = len(df)
total_weighted = df['MARSUPWT'].sum()
pos_weighted_cap = df.loc[df['CAP_VAL'] > 0, 'MARSUPWT'].sum()
pos_weighted_div = df.loc[df['DIV_VAL'] > 0, 'MARSUPWT'].sum()
pos_weighted_eitc = df.loc[df['SPM_EITC'] > 0, 'MARSUPWT'].sum()
pos_weighted_actc = df.loc[df['SPM_ACTC'] > 0, 'MARSUPWT'].sum()

print(f"\nSample size: {n_total}")
print(f"\nWeighted proportions:")
print(f"  CAP_VAL > 0: {pos_weighted_cap/total_weighted*100:.2f}%")
print(f"  DIV_VAL > 0: {pos_weighted_div/total_weighted*100:.2f}%")
print(f"  EITC  > 0:   {pos_weighted_eitc/total_weighted*100:.2f}%")
print(f"  ACTC  > 0:   {pos_weighted_actc/total_weighted*100:.2f}%")

# =============================================================================
# WEIGHTED RATIOS
# =============================================================================

total_agi_w = np.sum(df['MARSUPWT'] * df['AGI'])
total_eip_w = np.sum(df['MARSUPWT'] * df['filer_eip'])
total_eitc_w = np.sum(df['MARSUPWT'] * df['SPM_EITC'])
total_actc_w = np.sum(df['MARSUPWT'] * df['SPM_ACTC'])
total_eitc_actc_w = total_eitc_w + total_actc_w

ratio_eip = total_eip_w / total_agi_w
ratio_eitc_actc = total_eitc_actc_w / total_agi_w

print(f"\nWeighted ratios (relative to total AGI):")
print(f"  Total EIP / AGI:            {ratio_eip*100:.2f}%")
print(f"  Total (EITC + ACTC) / AGI:  {ratio_eitc_actc*100:.2f}%")
print(f"  Total EIP amount:           {total_eip_w:,.0f}")
print(f"  Total EITC+ACTC amount:     {total_eitc_actc_w:,.0f}")

# =============================================================================
# CONSTRUCT TAX VARIABLES
# =============================================================================

df['ltcg_estimate'] = df['CAP_VAL'] * LTCG_SHARE
df['qualified_div_estimate'] = df['DIV_VAL'] * QDIV_SHARE
df['pref_income'] = df['ltcg_estimate'] + df['qualified_div_estimate']

df['ordinary_income'] = np.clip(df['AGI'] - df['pref_income'], 0, None)
df['taxable_ordinary'] = np.clip(df['ordinary_income'] - STD_DEDUCTION, 0, None)

def calculate_preferential_tax(pref_income, agi):
    tax = np.zeros_like(pref_income)
    mask_15 = (agi > 80800) & (agi <= 501600)
    mask_20 = agi > 501600
    tax[mask_15] = pref_income[mask_15] * 0.15
    tax[mask_20] = pref_income[mask_20] * 0.20
    return tax

df['pref_tax'] = calculate_preferential_tax(df['pref_income'].values, df['AGI'].values)

df['niit_base'] = np.minimum(df['pref_income'], np.maximum(df['AGI'] - 250000, 0))
df['niit'] = np.where(df['AGI'] > 250000, df['niit_base'] * 0.038, 0)

# Extract arrays
agi = df['AGI'].values
eip_total = df['filer_eip'].values
taxable_ordinary = df['taxable_ordinary'].values
pref_tax = df['pref_tax'].values
niit = df['niit'].values
# eitc = df['SPM_EITC'].values
# actc = df['SPM_ACTC'].values

###see what happen without considering eitc and actc in tax year 2021
eitc = 0.0
actc = 0.0

fedtax_ac = df['FEDTAX_AC'].values

if USE_WEIGHTS:
    marsupwt = df['MARSUPWT'].values
else:
    marsupwt = np.ones_like(df['MARSUPWT'].values)

g_obs = np.sum(marsupwt * fedtax_ac) / np.sum(marsupwt * agi)
print(f"\nObserved g (FEDTAX_AC / AGI): {g_obs:.4f}")

# =============================================================================
# TAX FUNCTION
# =============================================================================

def compute_taxable_amounts(incomes, brackets):
    n = len(incomes)
    m = len(brackets) + 1
    taxable = np.zeros((n, m))
    remaining = incomes.astype(float).copy()
    taxable[:, 0] = np.minimum(remaining, brackets[0])
    remaining -= taxable[:, 0]
    for k in range(len(brackets) - 1):
        width = brackets[k + 1] - brackets[k]
        taxable[:, k + 1] = np.minimum(remaining, width)
        remaining -= taxable[:, k + 1]
    taxable[:, -1] = np.maximum(remaining, 0.0)
    return taxable

taxable = compute_taxable_amounts(taxable_ordinary, BRACKETS)

# =============================================================================
# OPTIMIZATION FUNCTIONS
# =============================================================================

x_stat = np.array([0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37])

def make_optimization_functions(eip_use):
    effective_weight = (marsupwt * agi) / np.sum(marsupwt * agi)

    def make_revenue_constraint(g):
        def revenue_constraint(x):
            return np.sum(marsupwt * (taxable @ x - eitc - actc - eip_use
                                      + pref_tax + niit)) \
                   - g * np.sum(marsupwt * agi)
        return revenue_constraint

    def optimize_tax_rates(sigma, g, x0=None):
        if x0 is None:
            x0 = x_stat.copy()

        constraints = [{'type': 'eq', 'fun': make_revenue_constraint(g)}]

        def objective_and_grad(x, sigma):
            y = agi - taxable @ x + eitc + actc + eip_use - pref_tax - niit
            y = np.maximum(y, 1e-6)

            if sigma == 1.0:
                util = np.dot(effective_weight, np.log(y))
                grad = -np.dot(effective_weight / y, taxable)
            else:
                util = np.dot(effective_weight, y**(1.0 - sigma) / (1.0 - sigma))
                grad = -np.dot(effective_weight * (y ** (-sigma)), taxable)

            return -util, -grad

        def objective_hess(x):
            y = agi - taxable @ x + eitc + actc + eip_use - pref_tax - niit
            y = np.maximum(y, 1e-6)

            if sigma == 1.0:
                second_deriv_weights = effective_weight / (y ** 2)
            else:
                second_deriv_weights = effective_weight * sigma * (y ** (-(sigma + 1)))

            return taxable.T @ (second_deriv_weights[:, None] * taxable)

        result = minimize(
            fun=lambda x: objective_and_grad(x, sigma),
            x0=x0,
            method='trust-constr',
            jac=True,
            hess=objective_hess,
            constraints=constraints,
            options=SOLVER_OPTIONS
        )
        return result

    return optimize_tax_rates

# =============================================================================
# BRACKET WEIGHTS (THREE CRITERIA)
# =============================================================================

ind_tax_actual = taxable @ x_stat
bracket_bounds = [0] + list(BRACKETS) + [np.inf]
n_brackets = len(BRACKETS) + 1

total_pop_weight = np.sum(marsupwt)
total_tax_weighted = np.sum(marsupwt * ind_tax_actual)

pi_k = np.zeros(n_brackets)
pi_k_tax = np.zeros(n_brackets)
pi_k_rev = np.zeros(n_brackets)

for k in range(n_brackets):
    lower = bracket_bounds[k]
    upper = bracket_bounds[k + 1]
    in_bracket = (taxable_ordinary >= lower) & (taxable_ordinary < upper)

    pi_k[k] = np.sum(marsupwt[in_bracket]) / total_pop_weight
    pi_k_tax[k] = np.sum(marsupwt[in_bracket] * ind_tax_actual[in_bracket]) / total_tax_weighted

    revenue_at_k = np.sum(marsupwt * taxable[:, k] * x_stat[k])
    pi_k_rev[k] = revenue_at_k / total_tax_weighted

print("\nBracket weights (three criteria):")
print(f"{'Bracket':>8} {'Population':>12} {'Tax paid':>12} {'Revenue':>12}")
print("-" * 50)
for k in range(n_brackets):
    print(f"{k+1:>8} {pi_k[k]:>12.4f} {pi_k_tax[k]:>12.4f} {pi_k_rev[k]:>12.4f}")
print("-" * 50)

def compute_waad(optimal_rates):
    rate_errors = np.abs(optimal_rates - x_stat)
    waad_pop = np.sum(pi_k * rate_errors) * 100
    waad_tax = np.sum(pi_k_tax * rate_errors) * 100
    waad_rev = np.sum(pi_k_rev * rate_errors) * 100
    return waad_pop, waad_tax, waad_rev

# =============================================================================
# RUN ESTIMATION FOR EACH SPECIFICATION
# =============================================================================

summary_results = []

for spec in SPECIFICATIONS:
    label = spec['label']
    include_eip = spec['include_eip']

    print("\n" + "="*100)
    print(f"SPECIFICATION: {label}")
    print("="*100)

    if include_eip:
        eip_use = eip_total.copy()
    else:
        eip_use = np.zeros_like(eip_total)

    optimize_func = make_optimization_functions(eip_use)

    best_pop = {'waad': np.inf, 'g': None, 'sigma': None,
                'waad_pop': None, 'waad_tax': None, 'waad_rev': None, 'rates': None}
    best_tax = {'waad': np.inf, 'g': None, 'sigma': None,
                'waad_pop': None, 'waad_tax': None, 'waad_rev': None, 'rates': None}
    best_rev = {'waad': np.inf, 'g': None, 'sigma': None,
                'waad_pop': None, 'waad_tax': None, 'waad_rev': None, 'rates': None}

    for g in G_GRID:
        print(f"\nProcessing g = {g:.3f}...")

        best_pop_g = {'waad': np.inf, 'g': g, 'sigma': None,
                      'waad_pop': None, 'waad_tax': None, 'waad_rev': None, 'rates': None}
        best_tax_g = {'waad': np.inf, 'g': g, 'sigma': None,
                      'waad_pop': None, 'waad_tax': None, 'waad_rev': None, 'rates': None}
        best_rev_g = {'waad': np.inf, 'g': g, 'sigma': None,
                      'waad_pop': None, 'waad_tax': None, 'waad_rev': None, 'rates': None}

        x0 = x_stat.copy()

        for sigma in SIGMA_GRID_01:
            result = optimize_func(sigma, g, x0=x0)

            if result.success:
                waad_pop, waad_tax, waad_rev = compute_waad(result.x)

                if waad_pop < best_pop_g['waad']:
                    best_pop_g.update({
                        'waad': waad_pop, 'sigma': sigma,
                        'waad_pop': waad_pop, 'waad_tax': waad_tax, 'waad_rev': waad_rev,
                        'rates': result.x.copy()
                    })

                if waad_tax < best_tax_g['waad']:
                    best_tax_g.update({
                        'waad': waad_tax, 'sigma': sigma,
                        'waad_pop': waad_pop, 'waad_tax': waad_tax, 'waad_rev': waad_rev,
                        'rates': result.x.copy()
                    })

                if waad_rev < best_rev_g['waad']:
                    best_rev_g.update({
                        'waad': waad_rev, 'sigma': sigma,
                        'waad_pop': waad_pop, 'waad_tax': waad_tax, 'waad_rev': waad_rev,
                        'rates': result.x.copy()
                    })

                x0 = result.x

        if best_pop_g['sigma'] is not None:
            print(f"  Best pop: sigma = {best_pop_g['sigma']:.3f}, "
                  f"WAAD_P = {best_pop_g['waad_pop']:.2f}%, WAAD_T = {best_pop_g['waad_tax']:.2f}%, WAAD_R = {best_pop_g['waad_rev']:.2f}%")
            print(f"  Best tax: sigma = {best_tax_g['sigma']:.3f}, "
                  f"WAAD_P = {best_tax_g['waad_pop']:.2f}%, WAAD_T = {best_tax_g['waad_tax']:.2f}%, WAAD_R = {best_tax_g['waad_rev']:.2f}%")
            print(f"  Best rev: sigma = {best_rev_g['sigma']:.3f}, "
                  f"WAAD_P = {best_rev_g['waad_pop']:.2f}%, WAAD_T = {best_rev_g['waad_tax']:.2f}%, WAAD_R = {best_rev_g['waad_rev']:.2f}%")

            if best_pop_g['waad'] < best_pop['waad']:
                best_pop.update(best_pop_g)
            if best_tax_g['waad'] < best_tax['waad']:
                best_tax.update(best_tax_g)
            if best_rev_g['waad'] < best_rev['waad']:
                best_rev.update(best_rev_g)

    summary_results.append({
        'label': label,
        'best_pop': best_pop,
        'best_tax': best_tax,
        'best_rev': best_rev,
    })

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\n" + "="*110)
print("SUMMARY: EIP TREATMENT (Three Criteria)")
print("="*110)
print(f"{'Specification':<20} {'Criterion':<12} {'g*':>8} {'sigma*':>10} {'WAAD_P':>9} {'WAAD_T':>9} {'WAAD_R':>9}")
print("-"*110)

for res in summary_results:
    label = res['label']
    for crit_name, crit in [('Population', res['best_pop']),
                            ('Tax-paid', res['best_tax']),
                            ('Revenue', res['best_rev'])]:
        print(f"{label:<20} {crit_name:<12} {crit['g']:>8.3f} {crit['sigma']:>10.3f} "
              f"{crit['waad_pop']:>8.2f}% {crit['waad_tax']:>8.2f}% {crit['waad_rev']:>8.2f}%")
    print("-"*110)
