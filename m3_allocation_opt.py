# ==============================================================================
# KHAI BÁO LIÊM CHÍNH HỌC THUẬT
# Sinh viên: Lưu Hồng Ngọc - Trường Đại học Kinh tế (UEB - VNU)
# MODULE 3: TỐI ƯU PHÂN BỔ (Tích hợp Bài 4 và Bài 8)
# ==============================================================================

import numpy as np
import pandas as pd
import pulp
from scipy.optimize import minimize

def optimize_regional_allocation(total_budget=50000):
    """
    Giải bài toán Bài 4: Quy hoạch tuyến tính phân bổ ngân sách cho 6 vùng.
    Đầu vào: Tổng ngân sách (mặc định 50.000 tỷ VND).
    Đầu ra: Ma trận phân bổ tối ưu (Dictionary).
    """
    regions = ['NMM', 'RRD', 'NCC', 'CH', 'SE', 'MD']
    items = ['I', 'D', 'AI', 'H']
    
    # Ma trận hệ số tác động beta
    beta = {
        ('NMM', 'I'): 1.15, ('NMM', 'D'): 0.85, ('NMM', 'AI'): 0.55, ('NMM', 'H'): 1.30,
        ('RRD', 'I'): 0.95, ('RRD', 'D'): 1.25, ('RRD', 'AI'): 1.40, ('RRD', 'H'): 1.05,
        ('NCC', 'I'): 1.05, ('NCC', 'D'): 0.95, ('NCC', 'AI'): 0.85, ('NCC', 'H'): 1.15,
        ('CH', 'I'): 1.20, ('CH', 'D'): 0.75, ('CH', 'AI'): 0.45, ('CH', 'H'): 1.35,
        ('SE', 'I'): 0.90, ('SE', 'D'): 1.30, ('SE', 'AI'): 1.55, ('SE', 'H'): 1.00,
        ('MD', 'I'): 1.10, ('MD', 'D'): 0.85, ('MD', 'AI'): 0.65, ('MD', 'H'): 1.25
    }
    
    # Khởi tạo bài toán tối đa hóa GDP
    m = pulp.LpProblem('VN_Digital_Budget', pulp.LpMaximize)
    x = pulp.LpVariable.dicts('x', (regions, items), lowBound=0)
    
    # Hàm mục tiêu
    m += pulp.lpSum(beta[(r, j)] * x[r][j] for r in regions for j in items)
    
    # Ràng buộc cơ bản
    m += pulp.lpSum(x[r][j] for r in regions for j in items) <= total_budget
    for r in regions:
        m += pulp.lpSum(x[r][j] for j in items) >= 5000
        m += pulp.lpSum(x[r][j] for j in items) <= 12000
    m += pulp.lpSum(x[r]['H'] for r in regions) >= 12000
    
    # Giải bài toán (tắt thông báo log để không làm rác Dashboard)
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Trích xuất kết quả
    results = {}
    for r in regions:
        results[r] = {j: pulp.value(x[r][j]) for j in items}
        
    return results, pulp.value(m.objective)

def optimize_dynamic_trajectory(K0=27500, D0=20.3, AI0=86, H0=30, T=10):
    """
    Giải bài toán Bài 8: Tối ưu phân bổ liên thời gian 2026-2035.
    Đầu vào: Vốn vật chất, hạ tầng số, AI và nhân lực ban đầu.
    Đầu ra: Quỹ đạo đầu tư và GDP tối ưu qua các năm.
    """
    # Định nghĩa hàm mục tiêu nội bộ
    def objective(I_flat):
        I = I_flat.reshape(4, T)
        K, D, AI, H = np.zeros(T+1), np.zeros(T+1), np.zeros(T+1), np.zeros(T+1)
        K[0], D[0], AI[0], H[0] = K0, D0, AI0, H0
        L_const = 54.0
        utility = 0.0
        
        for t in range(T):
            Y = (K[t]**0.33) * (L_const**0.42) * (D[t]**0.10) * (AI[t]**0.08) * (H[t]**0.07)
            C = Y - np.sum(I[:, t])
            if C <= 1e-3: return 1e12 
            utility -= (0.97**t) * np.log(C)
            
            # Cập nhật vốn
            K[t+1] = K[t]*(1-0.05) + I[0, t]
            D[t+1] = D[t]*(1-0.12) + I[1, t]
            AI[t+1] = AI[t]*(1-0.15) + I[2, t]
            H[t+1] = H[t] + 0.8*I[3, t]
            
        return utility

    # Khởi tạo điểm bắt đầu và giới hạn giải
    x0 = np.ones(4 * T) * 800.0
    bounds = [(0, 15000) for _ in range(4 * T)]
    
    # Giải
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, options={'maxiter': 1000, 'disp': False})
    
    # Trả về kết quả thô để Dashboard (M6) tự vẽ đồ thị
    optimal_investment = res.x.reshape(4, T)
    return optimal_investment
