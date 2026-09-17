"""
Robustness check: Head of Household filing status
Tax year 2023, Data: CPS ASEC 2024 (pppub24.csv)

Two specifications:
1. HoH, EITC/ACTC included
2. HoH, EITC/ACTC excluded

Search:
- Stage 1: Coarse grid at 0.005 increments
- Stage 2: Fine grid at 0.001 increments around all three coarse optima

Reports three WAAD criteria at each criterion's own optimum:
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
FILESTAT_FILTER = 4  # Head of Household

STD_DEDUCTION = 20800
BRACKETS = np.array([15700, 59850, 95350, 182100, 231250, 578100])

PREF_TAX_0_LIMIT = 59750
PREF_TAX_15_LIMIT = 523050
NIIT_THRESHOLD = 200000

LTCG_SHARE = 1.0
QDIV_SHARE = 0.75
USE_WEIGHTS = True

SOLVER_OPTIONS = {
    'gtol': 1e-8,
    'xtol': 1e-8,
    'barrier_tol': 1e-8,
    'maxiter': 1000,
    'verbose': 0
}

G_GRID_COARSE = np.round(np.arange(0.020, 0.181, 0.005), 3)
SIGMA_GRID_COARSE = np.round(np.arange(1.000, 1.301, 0.005), 3)

SPECIFICATIONS = [
    {'label': 'HoH, EITC/ACTC included', 'include_eitc_actc': True},
    {'label': 'HoH, EITC/ACTC excluded', 'include_eitc_actc': False},
]

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

# Select Head of Household
df0 = data[data['FILESTAT'] == FILESTAT_FILTER].copy()
df_hh = df0[df0['AGI'] > 0].copy()

print(f"HoH sample size: {len(df_hh)}")

# Handle multiple records per tax unit
if 'TAX_ID' in df_hh.columns:
    n_units = df_hh['TAX_ID'].nunique()
    print(f"Number of unique tax units: {n_units}")
    print(f"Records per tax unit: {len(df_hh) / n_units:.2f}")

    df_hh = df_hh.sort_values(['TAX_ID', 'AGI'], ascending=[True, False])
    df_hh['is_head'] = df_hh.groupby('TAX_ID')['AGI'].transform(
        lambda x: (x == x.max())
    )
    df_hh = df_hh[df_hh['is_head']].copy()
    print(f"After keeping head: {len(df_hh)}")

# =============================================================================
# CONSTRUCT TAX VARIABLES
# =============================================================================

df_hh['ltcg_estimate'] = df_hh['CAP_VAL'].values * LTCG_SHARE
df_hh['qualified_div_estimate'] = df_hh['DIV_VAL'].values * QDIV_SHARE
df_hh['pref_income'] = df_hh['ltcg_estimate'] + df_hh['qualified_div_estimate']

df_hh['ordinary_income'] = np.clip(df_hh['AGI'].values - df_hh['pref_income'].values, 0, None)
df_hh['taxable_ordinary'] = np.clip(df_hh['ordinary_income'].values - STD_DEDUCTION, 0, None)

def calculate_preferential_tax(pref_income, agi):
    tax = np.zeros_like(pref_income)
    mask_15 = (agi > PREF_TAX_0_LIMIT) & (agi <= PREF_TAX_15_LIMIT)
    mask_20 = agi > PREF_TAX_15_LIMIT
    tax[mask_15] = pref_income[mask_15] * 0.15
    tax[mask_20] = pref_income[mask_20] * 0.20
    return tax

df_hh['pref_tax'] = calculate_preferential_tax(
    df_hh['pref_income'].values, df_hh['AGI'].values
)

df_hh['niit_base'] = np.minimum(
    df_hh['pref_income'].values,
    np.maximum(df_hh['AGI'].values - NIIT_THRESHOLD, 0)
)
df_hh['niit'] = np.where(
    df_hh['AGI'].values > NIIT_THRESHOLD,
    df_hh['niit_base'].values * 0.038,
    0
)

agi = df_hh['AGI'].values
taxable_ordinary = df_hh['taxable_ordinary'].values
pref_tax = df_hh['pref_tax'].values
niit = df_hh['niit'].values
eitc = df_hh['SPM_EITC'].values
actc = df_hh['SPM_ACTC'].values
fedtax_ac = df_hh['FEDTAX_AC'].values

if USE_WEIGHTS:
    marsupwt = df_hh['MARSUPWT'].values
else:
    marsupwt = np.ones_like(df_hh['MARSUPWT'].values)

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

x_stat = np.array([0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37])

# =============================================================================
# OPTIMIZATION FUNCTIONS
# =============================================================================

def make_optimization_functions(eitc_use, actc_use):
    effective_weight = (marsupwt * agi) / np.sum(marsupwt * agi)

    def make_revenue_constraint(g):
        def revenue_constraint(x):
            return np.sum(marsupwt * (taxable @ x - eitc_use - actc_use + pref_tax + niit)) \
                   - g * np.sum(marsupwt * agi)
        return revenue_constraint

    def optimize_tax_rates(sigma, g, x0=None):
        if x0 is None:
            x0 = x_stat.copy()

        constraints = [{'type': 'eq', 'fun': make_revenue_constraint(g)}]

        def objective_and_grad(x, sigma):
            y = agi - taxable @ x + eitc_use + actc_use - pref_tax - niit
            y = np.maximum(y, 1e-6)

            if sigma == 1.0:
                util = np.dot(effective_weight, np.log(y))
                grad = -np.dot(effective_weight / y, taxable)
            else:
                util = np.dot(effective_weight, y**(1.0 - sigma) / (1.0 - sigma))
                grad = -np.dot(effective_weight * (y ** (-sigma)), taxable)

            return -util, -grad

        def objective_hess(x):
            y = agi - taxable @ x + eitc_use + actc_use - pref_tax - niit
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

assert np.isclose(np.sum(pi_k), 1.0)
assert np.isclose(np.sum(pi_k_tax), 1.0)
assert np.isclose(np.sum(pi_k_rev), 1.0)

def compute_waad(optimal_rates):
    """Compute all three WAAD values for a given rate vector."""
    rate_errors = np.abs(optimal_rates - x_stat)
    waad_pop = np.sum(pi_k * rate_errors) * 100
    waad_tax = np.sum(pi_k_tax * rate_errors) * 100
    waad_rev = np.sum(pi_k_rev * rate_errors) * 100
    return waad_pop, waad_tax, waad_rev

# =============================================================================
# SEARCH FUNCTION
# =============================================================================

def run_search(optimize_func, g_grid, sigma_grid):
    best_pop = {'waad': np.inf, 'g': None, 'sigma': None,
                'waad_pop': None, 'waad_tax': None, 'waad_rev': None, 'rates': None}
    best_tax = {'waad': np.inf, 'g': None, 'sigma': None,
                'waad_pop': None, 'waad_tax': None, 'waad_rev': None, 'rates': None}
    best_rev = {'waad': np.inf, 'g': None, 'sigma': None,
                'waad_pop': None, 'waad_tax': None, 'waad_rev': None, 'rates': None}

    x0 = x_stat.copy()

    for g in g_grid:
        for sigma in sigma_grid:
            result = optimize_func(sigma, g, x0=x0)

            if result.success:
                waad_pop, waad_tax, waad_rev = compute_waad(result.x)

                if waad_pop < best_pop['waad']:
                    best_pop.update({
                        'waad': waad_pop, 'g': g, 'sigma': sigma,
                        'waad_pop': waad_pop, 'waad_tax': waad_tax, 'waad_rev': waad_rev,
                        'rates': result.x.copy()
                    })

                if waad_tax < best_tax['waad']:
                    best_tax.update({
                        'waad': waad_tax, 'g': g, 'sigma': sigma,
                        'waad_pop': waad_pop, 'waad_tax': waad_tax, 'waad_rev': waad_rev,
                        'rates': result.x.copy()
                    })

                if waad_rev < best_rev['waad']:
                    best_rev.update({
                        'waad': waad_rev, 'g': g, 'sigma': sigma,
                        'waad_pop': waad_pop, 'waad_tax': waad_tax, 'waad_rev': waad_rev,
                        'rates': result.x.copy()
                    })

                x0 = result.x

    return best_pop, best_tax, best_rev

# =============================================================================
# RUN ESTIMATION
# =============================================================================

summary_results = []

for spec in SPECIFICATIONS:
    label = spec['label']
    include_eitc_actc = spec['include_eitc_actc']

    print("\n" + "="*100)
    print(f"SPECIFICATION: {label}")
    print("="*100)

    if include_eitc_actc:
        eitc_use = eitc.copy()
        actc_use = actc.copy()
    else:
        eitc_use = np.zeros_like(eitc)
        actc_use = np.zeros_like(actc)

    optimize_func = make_optimization_functions(eitc_use, actc_use)

    # Stage 1: Coarse search
    print("Stage 1: Coarse search...")
    best_pop_coarse, best_tax_coarse, best_rev_coarse = run_search(
        optimize_func, G_GRID_COARSE, SIGMA_GRID_COARSE
    )

    if best_pop_coarse['sigma'] is not None:
        print(f"  Coarse best pop: g = {best_pop_coarse['g']:.3f}, "
              f"sigma = {best_pop_coarse['sigma']:.3f}, WAAD_P = {best_pop_coarse['waad_pop']:.2f}%")
    if best_tax_coarse['sigma'] is not None:
        print(f"  Coarse best tax: g = {best_tax_coarse['g']:.3f}, "
              f"sigma = {best_tax_coarse['sigma']:.3f}, WAAD_T = {best_tax_coarse['waad_tax']:.2f}%")
    if best_rev_coarse['sigma'] is not None:
        print(f"  Coarse best rev: g = {best_rev_coarse['g']:.3f}, "
              f"sigma = {best_rev_coarse['sigma']:.3f}, WAAD_R = {best_rev_coarse['waad_rev']:.2f}%")

    # Stage 2: Fine search around all three coarse optima
    fine_search_regions = []

    if best_pop_coarse['sigma'] is not None:
        g_fine_pop = np.round(np.arange(best_pop_coarse['g'] - 0.01,
                                        best_pop_coarse['g'] + 0.011, 0.001), 3)
        sigma_fine_pop = np.round(np.arange(best_pop_coarse['sigma'] - 0.03,
                                            best_pop_coarse['sigma'] + 0.013, 0.001), 3)
        fine_search_regions.append((g_fine_pop, sigma_fine_pop))

    if best_tax_coarse['sigma'] is not None:
        g_fine_tax = np.round(np.arange(best_tax_coarse['g'] - 0.01,
                                        best_tax_coarse['g'] + 0.011, 0.001), 3)
        sigma_fine_tax = np.round(np.arange(best_tax_coarse['sigma'] - 0.01,
                                            best_tax_coarse['sigma'] + 0.011, 0.001), 3)
        fine_search_regions.append((g_fine_tax, sigma_fine_tax))

    if best_rev_coarse['sigma'] is not None:
        g_fine_rev = np.round(np.arange(best_rev_coarse['g'] - 0.01,
                                        best_rev_coarse['g'] + 0.011, 0.001), 3)
        sigma_fine_rev = np.round(np.arange(best_rev_coarse['sigma'] - 0.01,
                                            best_rev_coarse['sigma'] + 0.011, 0.001), 3)
        fine_search_regions.append((g_fine_rev, sigma_fine_rev))

    if fine_search_regions:
        g_fine_combined = np.unique(np.concatenate([r[0] for r in fine_search_regions]))
        sigma_fine_combined = np.unique(np.concatenate([r[1] for r in fine_search_regions]))

        g_fine_combined = g_fine_combined[(g_fine_combined >= 0.01) & (g_fine_combined <= 0.25)]
        sigma_fine_combined = sigma_fine_combined[(sigma_fine_combined >= 1.00) & (sigma_fine_combined <= 1.35)]

        print(f"\nStage 2: Fine search around all coarse optima")
        print(f"  Total optimizations: {len(g_fine_combined) * len(sigma_fine_combined)}")

        best_pop_fine, best_tax_fine, best_rev_fine = run_search(
            optimize_func, g_fine_combined, sigma_fine_combined
        )

        best_pop = best_pop_fine if best_pop_fine['waad'] < best_pop_coarse['waad'] else best_pop_coarse
        best_tax = best_tax_fine if best_tax_fine['waad'] < best_tax_coarse['waad'] else best_tax_coarse
        best_rev = best_rev_fine if best_rev_fine['waad'] < best_rev_coarse['waad'] else best_rev_coarse
    else:
        best_pop = best_pop_coarse
        best_tax = best_tax_coarse
        best_rev = best_rev_coarse

    # -------------------------------------------------------------------------
    # Re-evaluate each optimum to obtain the full WAAD vector
    # -------------------------------------------------------------------------
    def evaluate_optimum(opt):
        if opt['sigma'] is None:
            return None, None, None
        result = optimize_func(opt['sigma'], opt['g'], x0=x_stat.copy())
        if result.success:
            return compute_waad(result.x)
        return None, None, None

    pop_waads = evaluate_optimum(best_pop)
    tax_waads = evaluate_optimum(best_tax)
    rev_waads = evaluate_optimum(best_rev)

    summary_results.append({
        'Specification': label,

        'sigma_pop': best_pop['sigma'],
        'g_pop': best_pop['g'],
        'waad_pop_at_pop': pop_waads[0],
        'waad_tax_at_pop': pop_waads[1],
        'waad_rev_at_pop': pop_waads[2],

        'sigma_tax': best_tax['sigma'],
        'g_tax': best_tax['g'],
        'waad_pop_at_tax': tax_waads[0],
        'waad_tax_at_tax': tax_waads[1],
        'waad_rev_at_tax': tax_waads[2],

        'sigma_rev': best_rev['sigma'],
        'g_rev': best_rev['g'],
        'waad_pop_at_rev': rev_waads[0],
        'waad_tax_at_rev': rev_waads[1],
        'waad_rev_at_rev': rev_waads[2],
    })

    # Detailed print
    print(f"\n--- Population-weighted optimum ---")
    print(f"  g = {best_pop['g']:.3f}, sigma = {best_pop['sigma']:.3f}")
    print(f"  WAAD_P = {pop_waads[0]:.2f}%, WAAD_T = {pop_waads[1]:.2f}%, WAAD_R = {pop_waads[2]:.2f}%")
    print(f"  Rates = {np.round(best_pop['rates'], 3)}")

    print(f"\n--- Tax-paid-weighted optimum ---")
    print(f"  g = {best_tax['g']:.3f}, sigma = {best_tax['sigma']:.3f}")
    print(f"  WAAD_P = {tax_waads[0]:.2f}%, WAAD_T = {tax_waads[1]:.2f}%, WAAD_R = {tax_waads[2]:.2f}%")
    print(f"  Rates = {np.round(best_tax['rates'], 3)}")

    print(f"\n--- Revenue-weighted optimum ---")
    print(f"  g = {best_rev['g']:.3f}, sigma = {best_rev['sigma']:.3f}")
    print(f"  WAAD_P = {rev_waads[0]:.2f}%, WAAD_T = {rev_waads[1]:.2f}%, WAAD_R = {rev_waads[2]:.2f}%")
    print(f"  Rates = {np.round(best_rev['rates'], 3)}")

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\n" + "="*115)
print("SUMMARY: HEAD OF HOUSEHOLD ROBUSTNESS (Three Criteria)")
print("="*115)
print(f"{'Specification':<32} {'Criterion':<12} {'sigma':>8} {'g':>8} {'WAAD_P':>8} {'WAAD_T':>8} {'WAAD_R':>8}")
print("-"*115)

for row in summary_results:
    spec = row['Specification']

    print(f"{spec:<32} {'Population':<12} {row['sigma_pop']:>8.3f} {row['g_pop']:>8.3f} "
          f"{row['waad_pop_at_pop']:>7.2f}% {row['waad_tax_at_pop']:>7.2f}% {row['waad_rev_at_pop']:>7.2f}%")

    print(f"{'':<32} {'Tax-paid':<12} {row['sigma_tax']:>8.3f} {row['g_tax']:>8.3f} "
          f"{row['waad_pop_at_tax']:>7.2f}% {row['waad_tax_at_tax']:>7.2f}% {row['waad_rev_at_tax']:>7.2f}%")

    print(f"{'':<32} {'Revenue':<12} {row['sigma_rev']:>8.3f} {row['g_rev']:>8.3f} "
          f"{row['waad_pop_at_rev']:>7.2f}% {row['waad_tax_at_rev']:>7.2f}% {row['waad_rev_at_rev']:>7.2f}%")

    print("-"*115)

print("\nNote: g and sigma reported to three decimal places from the fine search.")
print("WAAD values are evaluated at the respective criterion's optimum.")
