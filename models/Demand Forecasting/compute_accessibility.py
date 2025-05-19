import numpy as np
import pandas as pd

def compute_accessibility(
    df: pd.DataFrame,
    res,
    alt_id: str = 'mode_four_kinds',
    time_col: str = 'travel_time',
    cost_col: str = 'travel_cost',
    person_vars: list = ['age','male','numvec','hhinc'],
    group_col: str = 'work'
):
    """
    Calculate Logsum accessibility for each individual and compare groups.

    Parameters:
    - df: DataFrame with individual-mode records, including:
        alt_id, time_col, cost_col, person_vars, and group_col.
    - res: Fitted pylogit MNL result (res.params contains coefficients).
    - alt_id: Column name for mode identifier.
    - time_col: Column name for travel time.
    - cost_col: Column name for travel cost.
    - person_vars: List of socio-demographic variable names.
    - group_col: Column name for grouping (e.g., 'work').

    Returns:
    - group_stats: DataFrame with mean, median, std of accessibility by group.
    - full_result: DataFrame of each individual's accessibility and group.
    """
    coefs = res.params  # coefficients from the fitted model

    # Map numeric alternatives to mode labels
    mode_map = {0: 'auto', 1: 'transit', 2: 'bike', 3: 'walk'}

    def compute_ut(row):
        alt = row[alt_id]
        mode = mode_map[alt]
        utility = 0.0
        # Add alternative-specific constant
        if alt != 0:
            asc_name = f"asc_{mode}"
            utility += coefs.get(asc_name, 0.0)
        # Add travel time effect
        time_name = f"time_{mode}"
        utility += coefs.get(time_name, 0.0) * row[time_col]
        # Add travel cost effect if available
        cost_name = f"cost_{mode}"
        if cost_name in coefs.index and pd.notna(row.get(cost_col, np.nan)):
            utility += coefs[cost_name] * row[cost_col]
        # Add socio-demographic interaction effects
        for var in person_vars:
            var_name = f"{var}_{mode}"
            if var_name in coefs.index and pd.notna(row[var]):
                utility += coefs[var_name] * row[var]
        return utility

    # Compute utility for each person-mode record
    df = df.copy()
    df['utility'] = df.apply(compute_ut, axis=1)

    # Pivot to build person × mode utility matrix
    util_matrix = (
        df
        .pivot(index='sampno', columns=alt_id, values='utility')
        .fillna(-np.inf)  # treat missing modes as -inf utility
    )

    # Calculate Logsum accessibility
    logsum = np.log(np.exp(util_matrix).sum(axis=1))
    logsum.name = 'accessibility'

    # Merge group information
    full_result = (
        logsum
        .reset_index()
        .merge(df[['sampno', group_col]].drop_duplicates(),
               on='sampno', how='left')
    )

    # Compute group-level statistics
    group_stats = (
        full_result
        .groupby(group_col)['accessibility']
        .agg(['mean', 'median', 'std'])
        .rename(index={0: 'no', 1: 'yes'})
    )

    return group_stats, full_result

# Example usage:
# group_stats, full_df = compute_accessibility(df, res, group_col='work')
# print(group_stats)