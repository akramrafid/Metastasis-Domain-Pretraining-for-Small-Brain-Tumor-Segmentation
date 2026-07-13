"""
Statistical testing utilities, supporting Wilcoxon signed-rank test for continuous metrics,
McNemar's test for paired binary outcomes, and Benjamini-Hochberg FDR correction.
"""
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.stats import wilcoxon, binom, chi2

def run_wilcoxon_test(arm_a_scores: List[float], arm_b_scores: List[float]) -> Dict[str, Any]:
    """Runs paired Wilcoxon signed-rank test between two sets of patient scores.
    
    Args:
        arm_a_scores: List of scores for Arm A.
        arm_b_scores: List of scores for Arm B.
        
    Returns:
        Dictionary with test statistic and p-value.
    """
    a = np.array(arm_a_scores)
    b = np.array(arm_b_scores)
    
    if np.array_equal(a, b):
        return {"statistic": 0.0, "p_value": 1.0, "note": "Arrays are identical"}
        
    diff = a - b
    if np.all(diff == 0.0):
        return {"statistic": 0.0, "p_value": 1.0, "note": "Differences are all zero"}
        
    try:
        stat, p_val = wilcoxon(a, b)
        return {"statistic": float(stat), "p_value": float(p_val)}
    except Exception as e:
        return {"statistic": 0.0, "p_value": 1.0, "error": str(e)}

def mcnemar_test_paired_binary(arm_a_outcomes: List[float], arm_b_outcomes: List[float]) -> Dict[str, Any]:
    """Runs McNemar's test for paired binary classification outcomes (e.g. detected / not-detected).
    
    Uses exact binomial test when the number of discordant pairs is < 25, 
    and chi-squared test with Edwards' continuity correction otherwise.
    
    Args:
        arm_a_outcomes: Binary outcomes for Arm A (0 or 1).
        arm_b_outcomes: Binary outcomes for Arm B (0 or 1).
        
    Returns:
        Dictionary with test statistic and p-value.
    """
    a = np.array(arm_a_outcomes)
    b = np.array(arm_b_outcomes)
    
    # Enforce strictly binary outcomes (0 or 1)
    unique_vals = np.unique(np.concatenate([a, b]))
    for val in unique_vals:
        assert val in [0.0, 1.0, 0, 1], f"McNemar's test requires strictly binary outcomes, but found value: {val}"
        
    # Build contingency table:
    #                   Arm B = 1   Arm B = 0
    # Arm A = 1           n_11        n_10
    # Arm A = 0           n_01        n_00
    n_11 = int(np.sum((a == 1.0) & (b == 1.0)))
    n_10 = int(np.sum((a == 1.0) & (b == 0.0)))
    n_01 = int(np.sum((a == 0.0) & (b == 1.0)))
    n_00 = int(np.sum((a == 0.0) & (b == 0.0)))
    
    table = [[n_11, n_10], [n_01, n_00]]
    discordant_sum = n_10 + n_01
    exact = (discordant_sum < 25)
    
    if discordant_sum == 0:
        return {"statistic": 0.0, "p_value": 1.0, "table": table, "exact": exact, "note": "No discordant pairs"}
        
    if exact:
        # Exact binomial test under H0: p=0.5
        k = min(n_10, n_01)
        p_val = 2.0 * binom.cdf(k, discordant_sum, 0.5)
        # Cap at 1.0 for two-tailed test limit cases
        p_val = min(1.0, p_val)
        return {"statistic": float(k), "p_value": float(p_val), "table": table, "exact": True}
    else:
        # Chi-squared test with Edwards' continuity correction
        chi2_stat = float((abs(n_10 - n_01) - 1.0) ** 2) / discordant_sum
        p_val = chi2.sf(chi2_stat, df=1)
        return {"statistic": chi2_stat, "p_value": float(p_val), "table": table, "exact": False}

def run_significance_test(scores_a: List[float], scores_b: List[float], metric_name: str) -> Dict[str, Any]:
    """Unified router that directs continuous metrics to Wilcoxon and binary metrics to McNemar's test.
    
    Automatically filters out cases where either A or B has a NaN score to preserve paired indexing.
    """
    # Align and drop NaNs to keep paired samples
    aligned_a = []
    aligned_b = []
    for x, y in zip(scores_a, scores_b):
        if x is not None and y is not None and not np.isnan(x) and not np.isnan(y):
            aligned_a.append(float(x))
            aligned_b.append(float(y))
            
    if not aligned_a:
        return {"statistic": 0.0, "p_value": 1.0, "error": "No valid paired cases remaining after NaN filtering"}
        
    a = np.array(aligned_a)
    b = np.array(aligned_b)
    
    # Automatically check if outcomes are strictly binary
    is_binary_a = np.all(np.isin(a, [0.0, 1.0]))
    is_binary_b = np.all(np.isin(b, [0.0, 1.0]))
    is_binary = is_binary_a and is_binary_b
    
    if is_binary:
        return mcnemar_test_paired_binary(aligned_a, aligned_b)
    else:
        assert metric_name != "detection_rate", f"Detection rate must be strictly binary, but received continuous values: {a.tolist()}"
        return run_wilcoxon_test(aligned_a, aligned_b)

def benjamini_hochberg_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """Applies Benjamini-Hochberg False Discovery Rate (FDR) correction to a list of p-values."""
    p_arr = np.array(p_values)
    n = len(p_arr)
    if n == 0:
        return [], []
        
    sort_idx = np.argsort(p_arr)
    sorted_p = p_arr[sort_idx]
    
    adjusted_p = np.zeros(n)
    prev_adj = 1.0
    for i in range(n - 1, -1, -1):
        adj = sorted_p[i] * n / (i + 1)
        adj = min(prev_adj, adj)
        adjusted_p[i] = adj
        prev_adj = adj
        
    unsort_idx = np.argsort(sort_idx)
    adjusted_p_original = adjusted_p[unsort_idx]
    significant = adjusted_p_original < alpha
    return adjusted_p_original.tolist(), significant.tolist()
