"""
Unit tests for statistical significance tests and multiple testing corrections.
"""
import pytest
from src.evaluation.statistical_tests import (
    run_wilcoxon_test,
    mcnemar_test_paired_binary,
    run_significance_test,
    benjamini_hochberg_correction
)

def test_wilcoxon_test() -> None:
    """Verifies Wilcoxon signed-rank test calculation and fallback behaviors."""
    # Paired inputs with clear differences (N = 10)
    a = [0.8, 0.9, 0.85, 0.95, 0.78, 0.82, 0.88, 0.84, 0.91, 0.79]
    b = [0.6, 0.7, 0.65, 0.75, 0.58, 0.62, 0.68, 0.64, 0.71, 0.59]
    res = run_wilcoxon_test(a, b)
    assert res["p_value"] < 0.05
    
    # Identical arrays
    res_ident = run_wilcoxon_test(a, a)
    assert res_ident["p_value"] == 1.0
    assert "identical" in res_ident.get("note", "")

def test_mcnemar_paired_binary() -> None:
    """Verifies McNemar's paired binary test contingency table and calculations."""
    # Outcomes:
    # Patient 0: A=1, B=1 (concordant positive)
    # Patient 1: A=1, B=0 (discordant)
    # Patient 2: A=0, B=1 (discordant)
    # Patient 3: A=0, B=0 (concordant negative)
    a = [1.0, 1.0, 0.0, 0.0]
    b = [1.0, 0.0, 1.0, 0.0]
    
    res = mcnemar_test_paired_binary(a, b)
    assert res["table"] == [[1, 1], [1, 1]]
    assert res["p_value"] == 1.0  # Equal discordants (1 vs 1)
    
    # Check invalid inputs (non-binary outcomes)
    invalid_a = [0.8, 1.0, 0.0, 0.0]
    with pytest.raises(AssertionError, match="requires strictly binary outcomes"):
        mcnemar_test_paired_binary(invalid_a, b)

def test_significance_router() -> None:
    """Verifies that run_significance_test correctly routes continuous/binary metrics."""
    # Continuous metric -> Wilcoxon
    a_cont = [0.8, 0.9, 0.85]
    b_cont = [0.6, 0.7, 0.65]
    res_cont = run_significance_test(a_cont, b_cont, "dice")
    assert "table" not in res_cont  # Wilcoxon doesn't have a contingency table
    
    # Binary metric -> McNemar
    a_bin = [1.0, 0.0, 1.0]
    b_bin = [1.0, 1.0, 0.0]
    res_bin = run_significance_test(a_bin, b_bin, "detection_rate")
    assert "table" in res_bin  # McNemar has a contingency table
    
    # Check that "detection_rate" with continuous values fails assertion
    with pytest.raises(AssertionError, match="Detection rate must be strictly binary"):
        run_significance_test(a_cont, b_cont, "detection_rate")

def test_benjamini_hochberg_correction() -> None:
    """Verifies that Benjamini-Hochberg FDR correction calculates correct adjusted p-values."""
    p_values = [0.01, 0.04, 0.03, 0.001, 0.50]
    adjusted, sig = benjamini_hochberg_correction(p_values, alpha=0.05)
    
    assert len(adjusted) == len(p_values)
    assert len(sig) == len(p_values)
    
    # The smallest p-value (0.001) should remain significant and have the smallest adjusted value
    assert sig[3] is True
    # The largest p-value (0.50) should not be significant
    assert sig[4] is False
    # Adjusted p-values must be ordered correctly relative to raw p-values
    assert adjusted[3] < adjusted[0] < adjusted[2] < adjusted[1] < adjusted[4]
