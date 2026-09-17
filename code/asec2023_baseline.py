"""
Baseline estimation for tax year 2022
Married Filing Jointly, EITC and ACTC included as endogenous transfers
Data: CPS ASEC 2023 (pppub23.csv)

Reports three WAAD criteria:
1. Population-weighted
2. Tax-paid-weighted
3. Revenue-weighted

Automates:
1. Joint search over g and sigma at 0.001 increments
2. Finds the global best for all three criteria
3. Reports detailed results for selected sigma values at a fixed g
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore', message='delta_grad == 0.0')

# =============================================================================
# SETTINGS
# =============================================================================

DATA_FILE = "pppub23.csv"           # 2023 ASEC, tax year 2022
FILING_STATUS_MAX = 4               # FILESTAT < 4 for married filing jointly

STD_DEDUCTION = 25900                # 2022 MFJ standard deduction
BRACKETS = np.array([20550, 83550, 178150, 340100, 431900, 647850])

# Preferential income calibration
LTCG_SHARE = 1.0
QDIV_SHARE = 0.75

# Use survey weights?
USE_WEIGHTS = True

# Grid search parameters
G_GRID = np.round(np.arange(0.135, 0.155, 0.001), 3)
SIGMA_GRID = np.round(np.arange(1.050, 1.130, 0.001), 3)

# Detailed output for a specific g
PRINT_DETAILED_G = 0.14
DETAILED_SIGMA_VALUES = [1.000, 1.050, 1.080, 1.100, 1.110, 1.120, 1.130, 1.150, 1.200, 1.300]

# Solver settings
SOLVER_OPTIONS = {
    'gtol': 1e-8,
    'xtol': 1e-8,
    'barrier_tol': 1e-8,
    'maxiter': 1000,
    'verbose': 0
}

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

print(f"Reading {DATA_FILE}...")

required_cols = [
    'FILESTAT', 'TAX_ID', 'AGI', 'CAP_VAL', 'DIV_VAL', 'TAX_INC',
    'FEDTAX_AC', 'FEDTAX_BC', 'SPM_EITC', 'SPM_ACTC',
    'SPM_CAPHOUSESUB', 'SPM_SNAPSUB', 'MARSUPWT'
]

df_full = pd.read_csv(DATA_FILE)
missing_cols = [col for col in required_cols if col not in df_full.columns]
if missing_cols:
    print(f"Warning: Missing columns: {missing_cols}")

available_cols = [col for col in required_cols if col in df_full.columns]
data = df_full[available_cols].copy()

# Select married filing jointly
df0 = data[data['FILESTAT'] < FILING_STATUS_MAX].copy()

# Identify primary filer and aggregate spouse information
df0 = df0.sort_values(['TAX_ID', 'AGI'], ascending=[True, False])
df0['is_filer'] = df0.groupby('TAX_ID')['AGI'].transform(
    lambda x: (x == x.max()) & (x >= 0)
)
df0['total_cap'] = df0.groupby('TAX_ID')['CAP_VAL'].transform('sum')
df0['total_div'] = df0.groupby('TAX_ID')['DIV_VAL'].transform('sum')
df0['filer_cap'] = df0['total_cap'] * df0['is_filer']
df0['filer_div'] = df0['total_div'] * df0['is_filer']

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
# CONSTRUCT TAX VARIABLES
# =============================================================================

df['ltcg_estimate'] = df['CAP_VAL'] * LTCG_SHARE
df['qualified_div_estimate'] = df['DIV_VAL'] * QDIV_SHARE
df['pref_income'] = df['ltcg_estimate'] + df['qualified_div_estimate']

df['ordinary_income'] = np.clip(df['AGI'] - df['pref_income'], 0, None)
df['taxable_ordinary'] = np.clip(df['ordinary_income'] - STD_DEDUCTION, 0, None)

# Preferential tax (2022 MFJ thresholds: 0% up to $83,350; 15% up to $517,200)
def calculate_preferential_tax(pref_income, agi):
    tax = np.zeros_like(pref_income)
    mask_15 = (agi > 83350) & (agi <= 517200)
    mask_20 = agi > 517200
    tax[mask_15] = pref_income[mask_15] * 0.15
    tax[mask_20] = pref_income[mask_20] * 0.20
    return tax

df['pref_tax'] = calculate_preferential_tax(df['pref_income'].values, df['AGI'].values)

df['niit_base'] = np.minimum(df['pref_income'], np.maximum(df['AGI'] - 250000, 0))
df['niit'] = np.where(df['AGI'] > 250000, df['niit_base'] * 0.038, 0)

agi = df['AGI'].values
taxable_ordinary = df['taxable_ordinary'].values
pref_tax = df['pref_tax'].values
niit = df['niit'].values
eitc = df['SPM_EITC'].values
actc = df['SPM_ACTC'].values
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
# OPTIMIZATION
# =============================================================================

effective_weight = (marsupwt * agi) / np.sum(marsupwt * agi)
n_rates = len(BRACKETS) + 1
x_stat = np.array([0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37])

def make_revenue_constraint(g):
    def revenue_constraint(x):
        return np.sum(marsupwt * (taxable @ x - eitc - actc + pref_tax + niit)) \
               - g * np.sum(marsupwt * agi)
    return revenue_constraint

def optimize_tax_rates(sigma, g, x0=None):
    if x0 is None:
        x0 = x_stat.copy()

    constraints = [{'type': 'eq', 'fun': make_revenue_constraint(g)}]

    def objective_and_grad(x, sigma):
        y = agi - taxable @ x + eitc + actc - pref_tax - niit
        y = np.maximum(y, 1e-6)

        if sigma == 1.0:
            util = np.dot(effective_weight, np.log(y))
            grad = -np.dot(effective_weight / y, taxable)
        else:
            util = np.dot(effective_weight, y**(1.0 - sigma) / (1.0 - sigma))
            grad = -np.dot(effective_weight * (y ** (-sigma)), taxable)

        return -util, -grad

    def objective_hess(x):
        y = agi - taxable @ x + eitc + actc - pref_tax - niit
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

assert np.isclose(np.sum(pi_k), 1.0)
assert np.isclose(np.sum(pi_k_tax), 1.0)
assert np.isclose(np.sum(pi_k_rev), 1.0)

def compute_waad(optimal_rates):
    rate_errors = np.abs(optimal_rates - x_stat)
    waad_pop = np.sum(pi_k * rate_errors) * 100
    waad_tax = np.sum(pi_k_tax * rate_errors) * 100
    waad_rev = np.sum(pi_k_rev * rate_errors) * 100
    return waad_pop, waad_tax, waad_rev

# =============================================================================
# PART 1: DETAILED RESULTS FOR g = PRINT_DETAILED_G
# =============================================================================

print("\n" + "="*100)
print(f"DETAILED RESULTS FOR g = {PRINT_DETAILED_G:.3f}")
print("="*100)

detailed_results = []
x0 = x_stat.copy()

for sigma in DETAILED_SIGMA_VALUES:
    result = optimize_tax_rates(sigma, PRINT_DETAILED_G, x0=x0)
    if result.success:
        waad_pop, waad_tax, waad_rev = compute_waad(result.x)
        detailed_results.append({
            'sigma': sigma,
            'waad_pop': waad_pop,
            'waad_tax': waad_tax,
            'waad_rev': waad_rev,
            'rates': result.x
        })
        x0 = result.x
    else:
        print(f"  Failed at sigma = {sigma:.3f}")

print(f"{'Sigma':>8} {'WAAD_P':>9} {'WAAD_T':>9} {'WAAD_R':>9} "
      f"{'B1':>7} {'B2':>7} {'B3':>7} {'B4':>7} {'B5':>7} {'B6':>7} {'B7':>7}")
print("-"*100)
for d in detailed_results:
    r = d['rates']
    print(f"{d['sigma']:8.3f} {d['waad_pop']:8.2f}% {d['waad_tax']:8.2f}% {d['waad_rev']:8.2f}% "
          f"{r[0]:7.3f} {r[1]:7.3f} {r[2]:7.3f} {r[3]:7.3f} "
          f"{r[4]:7.3f} {r[5]:7.3f} {r[6]:7.3f}")
print("-"*100)

# =============================================================================
# PART 2: JOINT SEARCH OVER g AND sigma
# =============================================================================

results = []

print("\n" + "="*80)
print("JOINT SEARCH OVER g AND sigma (Three Criteria)")
print("="*80)

for g in G_GRID:
    print(f"\nProcessing g = {g:.3f}...")

    best_pop = {'waad': np.inf, 'g': g, 'sigma': None,
                'waad_pop': None, 'waad_tax': None, 'waad_rev': None, 'rates': None}
    best_tax = {'waad': np.inf, 'g': g, 'sigma': None,
                'waad_pop': None, 'waad_tax': None, 'waad_rev': None, 'rates': None}
    best_rev = {'waad': np.inf, 'g': g, 'sigma': None,
                'waad_pop': None, 'waad_tax': None, 'waad_rev': None, 'rates': None}

    x0 = x_stat.copy()

    for sigma in SIGMA_GRID:
        result = optimize_tax_rates(sigma, g, x0=x0)

        if result.success:
            waad_pop, waad_tax, waad_rev = compute_waad(result.x)

            if waad_pop < best_pop['waad']:
                best_pop.update({
                    'waad': waad_pop, 'sigma': sigma,
                    'waad_pop': waad_pop, 'waad_tax': waad_tax, 'waad_rev': waad_rev,
                    'rates': result.x.copy()
                })

            if waad_tax < best_tax['waad']:
                best_tax.update({
                    'waad': waad_tax, 'sigma': sigma,
                    'waad_pop': waad_pop, 'waad_tax': waad_tax, 'waad_rev': waad_rev,
                    'rates': result.x.copy()
                })

            if waad_rev < best_rev['waad']:
                best_rev.update({
                    'waad': waad_rev, 'sigma': sigma,
                    'waad_pop': waad_pop, 'waad_tax': waad_tax, 'waad_rev': waad_rev,
                    'rates': result.x.copy()
                })

            x0 = result.x

    if best_pop['sigma'] is not None:
        results.append(best_pop)
        results.append(best_tax)
        results.append(best_rev)
        print(f"  Best pop: sigma = {best_pop['sigma']:.3f}, "
              f"WAAD_P = {best_pop['waad_pop']:.2f}%, WAAD_T = {best_pop['waad_tax']:.2f}%, WAAD_R = {best_pop['waad_rev']:.2f}%")
        print(f"  Best tax: sigma = {best_tax['sigma']:.3f}, "
              f"WAAD_P = {best_tax['waad_pop']:.2f}%, WAAD_T = {best_tax['waad_tax']:.2f}%, WAAD_R = {best_tax['waad_rev']:.2f}%")
        print(f"  Best rev: sigma = {best_rev['sigma']:.3f}, "
              f"WAAD_P = {best_rev['waad_pop']:.2f}%, WAAD_T = {best_rev['waad_tax']:.2f}%, WAAD_R = {best_rev['waad_rev']:.2f}%")
    else:
        print(f"  No successful optimization for g = {g:.3f}")

# =============================================================================
# GLOBAL BEST
# =============================================================================

if results:
    global_pop = min(results, key=lambda x: x['waad_pop'])
    global_tax = min(results, key=lambda x: x['waad_tax'])
    global_rev = min(results, key=lambda x: x['waad_rev'])

    print("\n" + "="*80)
    print("GLOBAL BEST RESULTS")
    print("="*80)

    print(f"\nPopulation-weighted criterion:")
    print(f"  g = {global_pop['g']:.3f}")
    print(f"  sigma = {global_pop['sigma']:.3f}")
    print(f"  WAAD_P = {global_pop['waad_pop']:.2f}%")
    print(f"  WAAD_T = {global_pop['waad_tax']:.2f}%")
    print(f"  WAAD_R = {global_pop['waad_rev']:.2f}%")
    print(f"  Rates = {np.round(global_pop['rates'], 3)}")

    print(f"\nTax-paid-weighted criterion:")
    print(f"  g = {global_tax['g']:.3f}")
    print(f"  sigma = {global_tax['sigma']:.3f}")
    print(f"  WAAD_P = {global_tax['waad_pop']:.2f}%")
    print(f"  WAAD_T = {global_tax['waad_tax']:.2f}%")
    print(f"  WAAD_R = {global_tax['waad_rev']:.2f}%")
    print(f"  Rates = {np.round(global_tax['rates'], 3)}")

    print(f"\nRevenue-weighted criterion:")
    print(f"  g = {global_rev['g']:.3f}")
    print(f"  sigma = {global_rev['sigma']:.3f}")
    print(f"  WAAD_P = {global_rev['waad_pop']:.2f}%")
    print(f"  WAAD_T = {global_rev['waad_tax']:.2f}%")
    print(f"  WAAD_R = {global_rev['waad_rev']:.2f}%")
    print(f"  Rates = {np.round(global_rev['rates'], 3)}")
