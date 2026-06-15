# ==============================================================================
# KHAI BÁO LIÊM CHÍNH HỌC THUẬT
# Sinh viên: Lưu Hồng Ngọc - Trường Đại học Kinh tế (UEB - VNU)
# MODULE 5: ĐÁNH GIÁ RỦI RO & ĐA MỤC TIÊU (Tích hợp Bài 7 và Bài 10)
# ==============================================================================

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import pyomo.environ as pyo

# ------------------------------------------------------------------------------
# PHẦN 1: TỐI ƯU ĐA MỤC TIÊU PARETO (BÀI 7)
# ------------------------------------------------------------------------------
class VietnamDigitalProblem(Problem):
    """
    Định nghĩa bài toán Tối ưu hóa đa mục tiêu với 4 mục tiêu xung đột:
    Tăng trưởng, Bao trùm (Gini), Môi trường (Phát thải), An ninh mạng.
    """
    def __init__(self):
        super().__init__(n_var=24, n_obj=4, n_ieq_constr=14, 
                         xl=np.zeros(24), xu=np.ones(24) * 12000)
        
        # Ma trận hệ số beta (Tác động biên)
        self.beta = np.array([
            [1.15, 0.85, 0.55, 1.30], [0.95, 1.25, 1.40, 1.05],
            [1.05, 0.95, 0.85, 1.15], [1.20, 0.75, 0.45, 1.35],
            [0.90, 1.30, 1.55, 1.00], [1.10, 0.85, 0.65, 1.25]
        ])
        # Hệ số phát thải và rủi ro an ninh
        self.e = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])
        self.rho = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])
        self.sig = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])

    def _evaluate(self, x, out, *args, **kwargs):
        pop_size = x.shape[0]
        X = x.reshape(pop_size, 6, 4)
        
        # Tính toán 4 hàm mục tiêu
        f1 = -np.sum(X * self.beta, axis=(1, 2)) # Minimize -GDP = Maximize GDP
        
        reg_budgets = np.sum(X, axis=2)
        mean_budgets = np.mean(reg_budgets, axis=1, keepdims=True)
        f2 = np.mean(np.abs(reg_budgets - mean_budgets), axis=1) # Bất bình đẳng Gini
        
        f3 = np.sum(self.e * (X[:, :, 0] + X[:, :, 2]), axis=1) # Phát thải
        f4 = np.sum(self.rho * X[:, :, 2] - self.sig * X[:, :, 3], axis=1) # Rủi ro an ninh ròng
        
        out["F"] = np.column_stack([f1, f2, f3, f4])
        
        # Ràng buộc
        g = np.zeros((pop_size, 14))
        g[:, 0] = np.sum(X, axis=(1, 2)) - 50000
        for i in range(6): g[:, 1+i] = 5000 - reg_budgets[:, i]
        for i in range(6): g[:, 7+i] = reg_budgets[:, i] - 12000
        g[:, 13] = 12000 - np.sum(X[:, :, 3], axis=1)
        
        out["G"] = g

def evaluate_pareto_front():
    """
    Chạy thuật toán NSGA-II để tìm tập nghiệm Pareto cho 4 mục tiêu [cite: 469-470].
    Đầu ra: Tập hợp các điểm trên đường biên Pareto (Dùng cho Dashboard M6).
    """
    problem = VietnamDigitalProblem()
    algorithm = NSGA2(pop_size=50) # Rút gọn pop_size để chạy nhanh trên web
    res = minimize(problem, algorithm, ('n_gen', 50), seed=42, verbose=False)
    
    # Chuẩn bị dữ liệu trả về
    pareto_data = pd.DataFrame(res.F, columns=['GDP_Gain_Neg', 'Inequality', 'Emissions', 'CyberRisk'])
    pareto_data['GDP_Gain'] = -pareto_data['GDP_Gain_Neg'] # Trả lại dấu dương cho GDP
    return pareto_data


# ------------------------------------------------------------------------------
# PHẦN 2: QUY HOẠCH NGẪU NHIÊN 2 GIAI ĐOẠN (BÀI 10)
# ------------------------------------------------------------------------------
def evaluate_stochastic_risks():
    """
    Giải bài toán phân bổ ngân sách dưới 4 kịch bản bất định [cite: 666-667].
    Đầu ra: Kết quả phân bổ Giai đoạn 1 (Here-and-now) và Giai đoạn 2 (Wait-and-see) [cite: 671-684].
    """
    m = pyo.ConcreteModel()
    m.J = pyo.Set(initialize=['I', 'D', 'AI', 'H'])
    m.S = pyo.Set(initialize=['s1', 's2', 's3', 's4'])
    
    m.p = pyo.Param(m.S, initialize={'s1': 0.30, 's2': 0.45, 's3': 0.20, 's4': 0.05})
    m.beta = pyo.Param(m.J, initialize={'I': 1.00, 'D': 1.10, 'AI': 1.25, 'H': 0.95})
    
    beta_s = { 
        ('s1', 'I'): 1.25, ('s1', 'D'): 1.35, ('s1', 'AI'): 1.55, ('s1', 'H'): 1.05,
        ('s2', 'I'): 1.00, ('s2', 'D'): 1.10, ('s2', 'AI'): 1.25, ('s2', 'H'): 0.95,
        ('s3', 'I'): 0.75, ('s3', 'D'): 0.85, ('s3', 'AI'): 0.90, ('s3', 'H'): 1.00,
        ('s4', 'I'): 0.40, ('s4', 'D'): 0.50, ('s4', 'AI'): 0.55, ('s4', 'H'): 1.10
    }
    m.beta_s = pyo.Param(m.S, m.J, initialize=beta_s)
    
    m.x = pyo.Var(m.J, within=pyo.NonNegativeReals)
    m.y = pyo.Var(m.S, m.J, within=pyo.NonNegativeReals)
    
    m.budget1 = pyo.Constraint(expr=sum(m.x[j] for j in m.J) <= 65000) [cite: 676-678]
    
    def budget2_rule(m, s):
        return sum(m.y[s, j] for j in m.J) <= 15000 [cite: 689-690]
    m.budget2 = pyo.Constraint(m.S, rule=budget2_rule)
    
    def ai_link_rule(m, s):
        return m.y[s, 'AI'] <= 0.5 * m.x['H'] [cite: 694-695]
    m.ai_link = pyo.Constraint(m.S, rule=ai_link_rule)
    
    def obj_rule(m):
        first_stage = sum(m.beta[j] * m.x[j] for j in m.J)
        second_stage = sum(m.p[s] * sum(m.beta_s[s, j] * m.y[s, j] for j in m.J) for s in m.S)
        return first_stage + second_stage
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)
    
    solver = pyo.SolverFactory('glpk')
    solver.solve(m)
    
    # Trích xuất dữ liệu trả về cho Dashboard
    stage1_results = {j: pyo.value(m.x[j]) for j in m.J}
    return stage1_results, pyo.value(m.obj)
