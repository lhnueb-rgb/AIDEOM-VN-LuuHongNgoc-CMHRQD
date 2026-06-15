# ==============================================================================
# KHAI BÁO LIÊM CHÍNH HỌC THUẬT
# Sinh viên: Lưu Hồng Ngọc - Trường Đại học Kinh tế (UEB - VNU)
# MODULE 4: MÔ PHỎNG THỊ TRƯỜNG LAO ĐỘNG (BÀI 9)
# ==============================================================================

import numpy as np
import cvxpy as cp
import pandas as pd

def optimize_labor_impact(total_budget=30000):
    """
    Tối ưu hóa phân bổ đầu tư AI và Nhân lực (H) để tối đa hóa NetJob ròng.
    Hàm này mô phỏng sự dịch chuyển việc làm do tác động của tự động hóa 
    và năng lực đào tạo lại của nền kinh tế.
    
    Đầu vào: 
        - total_budget: Ngân sách tổng cho 8 ngành (mặc định 30.000 tỷ VND).
        
    Đầu ra: 
        - results (DataFrame): Bảng kết quả phân bổ và biến động việc làm.
        - total_netjob (float): Tổng số việc làm ròng tạo ra toàn nền kinh tế.
    """
    # Tên 8 ngành kinh tế
    sectors = [
        'Nông-Lâm-Thủy sản', 'CN chế biến chế tạo', 'Xây dựng',
        'Bán buôn bán lẻ', 'Tài chính-Ngân hàng', 'Logistics-Vận tải',
        'CNTT-Truyền thông', 'Giáo dục-Đào tạo'
    ]
    N = len(sectors)

    # Tham số thị trường lao động (từ Bảng 9.3)
    risk = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100.0
    a1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
    b1 = np.array([45.0, 28.0, 35.0, 32.0, 22.0, 30.0, 20.0, 55.0])
    c1 = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
    d1 = np.array([50.0, 32.0, 42.0, 38.0, 26.0, 36.0, 24.0, 62.0])

    # Khai báo biến quyết định: Đầu tư vào AI và Đào tạo (H)
    x_AI = cp.Variable(N, nonneg=True)
    x_H = cp.Variable(N, nonneg=True)

    # Các phương trình động lực học việc làm
    NewJob = cp.multiply(a1, x_AI)
    Upgrade = cp.multiply(b1, x_H)
    Displaced = cp.multiply(cp.multiply(c1, risk), x_AI)
    RetrainCap = cp.multiply(d1, x_H)

    # Việc làm ròng = Việc mới + Việc nâng cấp - Việc bị thay thế
    NetJob = NewJob + Upgrade - Displaced

    # Hệ thống ràng buộc
    constraints = [
        cp.sum(x_AI + x_H) <= total_budget,
        NetJob >= 0,                     # An sinh xã hội: Không ngành nào âm NetJob
        Displaced <= RetrainCap          # Năng lực đào tạo phải bao phủ số lao động dôi dư
    ]

    # Khởi tạo và giải bài toán
    prob = cp.Problem(cp.Maximize(cp.sum(NetJob)), constraints)
    prob.solve(solver=cp.ECOS)

    # Trích xuất kết quả ra DataFrame để hiển thị trên Dashboard
    results = pd.DataFrame({
        'Ngành': sectors,
        'Đầu tư AI (Tỷ VND)': np.round(x_AI.value, 2),
        'Đầu tư H (Tỷ VND)': np.round(x_H.value, 2),
        'Việc mới tạo ra': np.round(NewJob.value, 0),
        'Việc làm mất đi': np.round(Displaced.value, 0),
        'NetJob (Ròng)': np.round(NetJob.value, 0)
    })

    return results, prob.value
