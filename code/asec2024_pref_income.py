"""
Robustness check: Alternative treatments of preferential income
Tax year 2023, Married Filing Jointly, EITC and ACTC included

Five specifications:
1. 100% LTCG, 75% QDIV  (baseline)
2. 100% LTCG, 100% QDIV
3. 75% LTCG, 85% QDIV
4. 0% LTCG, 0% QDIV
5. 0% LTCG, 0% QDIV, using TAX_INC directly as the tax base

For each specification:
- Run joint search over g and sigma at 0.001 increments
- Find the global best for all three WAAD criteria
- Report the results in a summary table

Three criteria:
1. Population-weighted
2. Tax-paid-weighted
3. Revenue-weighted
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore', message='delta_grad == 0.0')

# =============================================================================
# SETTINGS
# =============================================================================

DATA_FILE = "pppub24.csv"
FILING_STATUS_MAX = 4

STD_DEDUCTION = 27700
BRACKETS = np.array([22000, 89450, 190750, 364200, 462500, 693750])

USE_WEIGHTS = True

# Grid search parameters
G_GRID = np.round(np.arange(0.120, 0.150, 0.001), 3)
SIGMA_GRID = np.round(np.arange(1.005, 1.135, 0.001), 3)

SOLVER_OPTIONS = {
    'gtol': 1e-8,
    'xtol': 1e-8,
    'barrier_tol': 1e-8,
    'maxiter': 1000,
    'verbose': 0
}

# =============================================================================
# SPECIFICATIONS TO RUN
# =============================================================================

SPECIFICATIONS = [
    {
        'label': '100% LTCG, 75% QDIV',
        'ltcg_share': 1.0,
        'qdiv_share': 0.75,
        'use_tax_inc': False,
    },
    {
        'label': '100% LTCG, 100% QDIV',
        'ltcg_share': 1.0,
        'qdiv_share': 1.0,
        'use_tax_inc': False,
    },
    {
        'label': '75% LTCG, 85% QDIV',
        'ltcg_share': 0.75,
        'qdiv_share': 0.85,
        'use_tax_inc': False,
    },
    {
        'label': '0% LTCG, 0% QDIV',
        'ltcg_share': 0.0,
        'qdiv_share': 0.0,
        'use_tax_inc': False,
    },
    {
        'label': '0% LTCG, 0% QDIV, using TAX_INC',
        'ltcg_share': 0.0,
        'qdiv_share': 0.0,
        'use_tax_inc': True,
    },
]

# =============================================================================
# LOAD AND PREPARE DATA (common to all specifications)
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

# Extract common arrays
agi = df['AGI'].values
eitc = df['SPM_EITC'].values
actc = df['SPM_ACTC'].values
fedtax_ac = df['FEDTAX_AC'].values
tax_inc = df['TAX_INC'].values

if USE_WEIGHTS:
    marsupwt = df['MARSUPWT'].values
else:
    marsupwt = np.ones_like(df['MARSUPWT'].values)

print(f"\nSample size: {len(df)}")
print(f"USE_WEIGHTS = {USE_WEIGHTS}")

# =============================================================================
# COMMON FUNCTIONS
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

def calculate_preferential_tax(pref_income, agi_in):
    tax = np.zeros_like(pref_income)
    mask_15 = (agi_in > 89250) & (agi_in <= 553850)
    mask_20 = agi_in > 553850
    tax[mask_15] = pref_income[mask_15] * 0.15
    tax[mask_20] = pref_income[mask_20] * 0.20
    return tax

# =============================================================================
# RUN ESTIMATION FOR EACH SPECIFICATION
# =============================================================================

summary_results = []

for spec in SPECIFICATIONS:
    label = spec['label']
    ltcg_share = spec['ltcg_share']
    qdiv_share = spec['qdiv_share']
    use_tax_inc = spec['use_tax_inc']

    print("\n" + "="*80)
    print(f"SPECIFICATION: {label}")
    print("="*80)

    # Construct specification-specific variables
    if use_tax_inc:
        taxable_ordinary = np.maximum(tax_inc, 0)
        pref_income = np.zeros_like(agi)
        pref_tax = np.zeros_like(agi)
        niit = np.zeros_like(agi)
    else:
        ltcg_estimate = df['CAP_VAL'].values * ltcg_share
        qdiv_estimate = df['DIV_VAL'].values * qdiv_share
        pref_income = ltcg_estimate + qdiv_estimate

        ordinary_income = np.clip(agi - pref_income, 0, None)
        taxable_ordinary = np.clip(ordinary_income - STD_DEDUCTION, 0, None)

        pref_tax = calculate_preferential_tax(pref_income, agi)
        niit_base = np.minimum(pref_income, np.maximum(agi - 250000, 0))
        niit = np.where(agi > 250000, niit_base * 0.038, 0)

    taxable = compute_taxable_amounts(taxable_ordinary, BRACKETS)

    effective_weight = (marsupwt * agi) / np.sum(marsupwt * agi)

    x_stat = np.array([0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37])
    n_rates = len(BRACKETS) + 1

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

    def compute_waad(optimal_rates):
        rate_errors = np.abs(optimal_rates - x_stat)
        waad_pop = np.sum(pi_k * rate_errors) * 100
        waad_tax = np.sum(pi_k_tax * rate_errors) * 100
        waad_rev = np.sum(pi_k_rev * rate_errors) * 100
        return waad_pop, waad_tax, waad_rev

    # Joint search
    best_pop = {'waad': np.inf, 'g': None, 'sigma': None,
                'waad_pop': None, 'waad_tax': None, 'waad_rev': None, 'rates': None}
    best_tax = {'waad': np.inf, 'g': None, 'sigma': None,
                'waad_pop': None, 'waad_tax': None, 'waad_rev': None, 'rates': None}
    best_rev = {'waad': np.inf, 'g': None, 'sigma': None,
                'waad_pop': None, 'waad_tax': None, 'waad_rev': None, 'rates': None}

    for g in G_GRID:
        x0 = x_stat.copy()

        best_pop_for_g = {'waad': np.inf, 'g': g, 'sigma': None,
                          'waad_pop': None, 'waad_tax': None, 'waad_rev': None, 'rates': None}
        best_tax_for_g = {'waad': np.inf, 'g': g, 'sigma': None,
                          'waad_pop': None, 'waad_tax': None, 'waad_rev': None, 'rates': None}
        best_rev_for_g = {'waad': np.inf, 'g': g, 'sigma': None,
                          'waad_pop': None, 'waad_tax': None, 'waad_rev': None, 'rates': None}

        for sigma in SIGMA_GRID:
            result = optimize_tax_rates(sigma, g, x0=x0)

            if result.success:
                waad_pop, waad_tax, waad_rev = compute_waad(result.x)

                if waad_pop < best_pop_for_g['waad']:
                    best_pop_for_g.update({
                        'waad': waad_pop, 'sigma': sigma,
                        'waad_pop': waad_pop, 'waad_tax': waad_tax, 'waad_rev': waad_rev,
                        'rates': result.x.copy()
                    })

                if waad_tax < best_tax_for_g['waad']:
                    best_tax_for_g.update({
                        'waad': waad_tax, 'sigma': sigma,
                        'waad_pop': waad_pop, 'waad_tax': waad_tax, 'waad_rev': waad_rev,
                        'rates': result.x.copy()
                    })

                if waad_rev < best_rev_for_g['waad']:
                    best_rev_for_g.update({
                        'waad': waad_rev, 'sigma': sigma,
                        'waad_pop': waad_pop, 'waad_tax': waad_tax, 'waad_rev': waad_rev,
                        'rates': result.x.copy()
                    })

                x0 = result.x

        if best_pop_for_g['waad'] < best_pop['waad']:
            best_pop.update(best_pop_for_g)

        if best_tax_for_g['waad'] < best_tax['waad']:
            best_tax.update(best_tax_for_g)

        if best_rev_for_g['waad'] < best_rev['waad']:
            best_rev.update(best_rev_for_g)

    # Store results
    summary_results.append({
        'Specification': label,

        'sigma_pop': best_pop['sigma'],
        'g_pop': best_pop['g'],
        'waad_pop_at_pop': best_pop['waad_pop'],
        'waad_tax_at_pop': best_pop['waad_tax'],
        'waad_rev_at_pop': best_pop['waad_rev'],

        'sigma_tax': best_tax['sigma'],
        'g_tax': best_tax['g'],
        'waad_pop_at_tax': best_tax['waad_pop'],
        'waad_tax_at_tax': best_tax['waad_tax'],
        'waad_rev_at_tax': best_tax['waad_rev'],

        'sigma_rev': best_rev['sigma'],
        'g_rev': best_rev['g'],
        'waad_pop_at_rev': best_rev['waad_pop'],
        'waad_tax_at_rev': best_rev['waad_tax'],
        'waad_rev_at_rev': best_rev['waad_rev'],
    })

    print(f"\nGlobal best (population-weighted):")
    print(f"  g = {best_pop['g']:.3f}")
    print(f"  sigma = {best_pop['sigma']:.3f}")
    print(f"  WAAD_P = {best_pop['waad_pop']:.2f}%")
    print(f"  WAAD_T = {best_pop['waad_tax']:.2f}%")
    print(f"  WAAD_R = {best_pop['waad_rev']:.2f}%")
    print(f"  Rates = {np.round(best_pop['rates'], 3)}")

    print(f"\nGlobal best (tax-paid-weighted):")
    print(f"  g = {best_tax['g']:.3f}")
    print(f"  sigma = {best_tax['sigma']:.3f}")
    print(f"  WAAD_P = {best_tax['waad_pop']:.2f}%")
    print(f"  WAAD_T = {best_tax['waad_tax']:.2f}%")
    print(f"  WAAD_R = {best_tax['waad_rev']:.2f}%")
    print(f"  Rates = {np.round(best_tax['rates'], 3)}")

    print(f"\nGlobal best (revenue-weighted):")
    print(f"  g = {best_rev['g']:.3f}")
    print(f"  sigma = {best_rev['sigma']:.3f}")
    print(f"  WAAD_P = {best_rev['waad_pop']:.2f}%")
    print(f"  WAAD_T = {best_rev['waad_tax']:.2f}%")
    print(f"  WAAD_R = {best_rev['waad_rev']:.2f}%")
    print(f"  Rates = {np.round(best_rev['rates'], 3)}")

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\n" + "="*140)
print("SUMMARY: ROBUSTNESS TO PREFERENTIAL INCOME ASSUMPTIONS (Three Criteria)")
print("="*140)
print(f"{'Specification':<36} {'Criterion':<12} {'sigma':>8} {'g':>8} {'WAAD_P':>8} {'WAAD_T':>8} {'WAAD_R':>8}")
print("-"*140)

for row in summary_results:
    spec = row['Specification']

    print(f"{spec:<36} {'Population':<12} {row['sigma_pop']:>8.3f} {row['g_pop']:>8.3f} "
          f"{row['waad_pop_at_pop']:>7.2f}% {row['waad_tax_at_pop']:>7.2f}% {row['waad_rev_at_pop']:>7.2f}%")
    print(f"{'':<36} {'Tax-paid':<12} {row['sigma_tax']:>8.3f} {row['g_tax']:>8.3f} "
          f"{row['waad_pop_at_tax']:>7.2f}% {row['waad_tax_at_tax']:>7.2f}% {row['waad_rev_at_tax']:>7.2f}%")
    print(f"{'':<36} {'Revenue':<12} {row['sigma_rev']:>8.3f} {row['g_rev']:>8.3f} "
          f"{row['waad_pop_at_rev']:>7.2f}% {row['waad_tax_at_rev']:>7.2f}% {row['waad_rev_at_rev']:>7.2f}%")
    print("-"*140)

print("\nNote: g and sigma reported to three decimal places from the joint search.")
print("WAAD values are evaluated at the respective criterion's optimum.")
