import pytest
import numpy as np
from m1_macro_forecast import calculate_tfp
from m2_digital_readiness import calculate_topsis
from m3_allocation_opt import optimize_regional_allocation
from m4_labor_sim import optimize_labor_impact
from m5_risk_eval import evaluate_stochastic_risks

def test_tfp_calculation():
    Y, K, L, D, AI, H = np.array([100]), np.array([10]), np.array([5]), np.array([2]), np.array([1]), np.array([1])
    A = calculate_tfp(Y, K, L, D, AI, H)
    assert A[0] > 0

def test_topsis_dimensions():
    X = np.array([[10, 5], [20, 2]])
    w = np.array([0.5, 0.5])
    flags = np.array([True, True])
    scores = calculate_topsis(X, w, flags)
    assert len(scores) == 2

def test_allocation_opt():
    # Kiểm tra xem module 3 có trả về đúng cấu trúc kết quả không
    results, obj_val = optimize_regional_allocation(total_budget=1000)
    assert 'NMM' in results
    assert obj_val >= 0

def test_labor_sim():
    # Kiểm tra xem module 4 có trả về đúng format không
    df, total_netjob = optimize_labor_impact(total_budget=1000)
    assert len(df) == 8 # Kiểm tra đủ 8 ngành

def test_stochastic_risk():
    # Kiểm tra xem module 5 có trả về kết quả
    results, obj_val = evaluate_stochastic_risks()
    assert 'AI' in results
    assert obj_val is not None
