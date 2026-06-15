import numpy as np
import pandas as pd

def calculate_topsis(X_matrix, weights, benefit_flags):
    """
    Tính toán điểm số TOPSIS để xếp hạng các phương án (vùng kinh tế).
    Đầu vào:
        - X_matrix: Ma trận dữ liệu đầu vào (numpy array).
        - weights: Mảng trọng số của từng tiêu chí.
        - benefit_flags: Mảng boolean (True nếu là tiêu chí lợi ích, False nếu là tiêu chí chi phí).
    Đầu ra:
        - C_star: Mảng điểm số gần gũi lý tưởng (hệ số từ 0 đến 1).
    """
    # 1. Chuẩn hóa ma trận
    norm_factor = np.sqrt((X_matrix**2).sum(axis=0))
    R = X_matrix / norm_factor
    
    # 2. Nhân trọng số
    V = R * weights
    
    # 3. Xác định giải pháp lý tưởng dương và âm
    A_star = np.where(benefit_flags, V.max(axis=0), V.min(axis=0))
    A_neg = np.where(benefit_flags, V.min(axis=0), V.max(axis=0))
    
    # 4. Tính khoảng cách
    S_star = np.sqrt(((V - A_star)**2).sum(axis=1))
    S_neg = np.sqrt(((V - A_neg)**2).sum(axis=1))
    
    # 5. Tính hệ số gần gũi
    C_star = S_neg / (S_star + S_neg)
    return C_star

def calculate_entropy_weights(X_matrix):
    """
    Tính toán trọng số khách quan bằng phương pháp Entropy.
    Đầu vào: Ma trận dữ liệu X_matrix.
    Đầu ra: Mảng trọng số khách quan (có tổng bằng 1).
    """
    P = X_matrix / X_matrix.sum(axis=0)
    k = 1.0 / np.log(len(X_matrix))
    E = -k * np.sum(P * np.log(P + 1e-12), axis=0)
    d = 1 - E
    return d / d.sum()
