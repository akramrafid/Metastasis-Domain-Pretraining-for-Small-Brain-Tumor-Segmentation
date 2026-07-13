"""
Statistical tests (paired Wilcoxon signed-rank test comparing Arms).
"""
from typing import List, Dict, Any
import numpy as np
from scipy.stats import wilcoxon

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
    
    # Check if they are identical
    if np.array_equal(a, b):
        return {"statistic": 0.0, "p_value": 1.0, "note": "Arrays are identical"}
        
    # Check if all differences are zero
    diff = a - b
    if np.all(diff == 0.0):
        return {"statistic": 0.0, "p_value": 1.0, "note": "Differences are all zero"}
        
    try:
        stat, p_val = wilcoxon(a, b)
        return {"statistic": float(stat), "p_value": float(p_val)}
    except Exception as e:
        return {"statistic": 0.0, "p_value": 1.0, "error": str(e)}
