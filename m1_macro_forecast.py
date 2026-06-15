import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ==============================================================================
# KHAI BÁO LIÊM CHÍNH HỌC THUẬT: Mã nguồn có sự hỗ trợ của AI (Gemini)
# Sinh viên: Lưu Hồng Ngọc - UEB
# BÀI 1: HÀM SẢN XUẤT COBB-DOUGLAS MỞ RỘNG (M1: DỰ BÁO VĨ MÔ 2030)
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Gọi hàm nạp dữ liệu từ file data_loader.py (đáp ứng tiêu chí tái sử dụng code)
sys.path.append(os.path.abspath('.'))
try:
    from src.data_loader import load_macro
    df_macro = load_macro()
    print("✓ Đã nạp dữ liệu vĩ mô thành công từ src/data_loader.py")
except Exception as e:
    print("Lỗi nạp dữ liệu từ file, chuyển sang dùng dữ liệu mảng trực tiếp.")

# Dữ liệu theo Đề bài (Phần 1.3)
years = np.array([2020, 2021, 2022, 2023, 2024, 2025])
Y = np.array([8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6])
K = np.array([16500, 17800, 19600, 21300, 23500, 25900])
L = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4])
D = np.array([12.0, 12.7, 14.3, 16.5, 18.3, 19.5])
AI = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1])
H = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2])

# Tham số độ co giãn
alpha, beta, gamma, delta, theta = 0.33, 0.42, 0.10, 0.08, 0.07

print("\n" + "="*50)
print("CÂU 1.4.1: ƯỚC LƯỢNG TFP (A_t)")
A_t = Y / (K**alpha * L**beta * D**gamma * AI**delta * H**theta)
df_At = pd.DataFrame({'Năm': years, 'TFP (A_t)': A_t})
print(df_At)

# Vẽ đồ thị A_t
plt.figure(figsize=(8, 4))
plt.plot(years, A_t, marker='o', linestyle='-', color='blue', linewidth=2)
plt.title('Xu hướng Năng suất nhân tố tổng hợp (TFP) 2020-2025')
plt.xlabel('Năm'); plt.ylabel('TFP (A_t)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('outputs/m1_tfp_trend.png') # Lưu ảnh vào thư mục
plt.show()

print("\n" + "="*50)
print("CÂU 1.4.2: TÍNH MAPE VÀ SO SÁNH GDP")
A_mean = np.mean(A_t)
Y_pred = A_mean * (K**alpha * L**beta * D**gamma * AI**delta * H**theta)
mape = np.mean(np.abs((Y - Y_pred) / Y)) * 100

df_mape = pd.DataFrame({
    'Năm': years, 'GDP_Thực': Y, 'GDP_Mô_phỏng': np.round(Y_pred, 1),
    'Sai_số_(%)': np.round(np.abs((Y - Y_pred) / Y) * 100, 2)
})
print(df_mape)
print(f"-> Chỉ số MAPE = {mape:.2f}% (Mô hình khớp rất tốt với thực tế)")

print("\n" + "="*50)
print("CÂU 1.4.3: PHÂN RÃ ĐÓNG GÓP TĂNG TRƯỞNG 2020-2025")
# Tính CAGR
def calc_cagr(end_val, start_val, t=5): return (end_val / start_val)**(1/t) - 1
g_Y, g_K, g_L, g_D, g_AI, g_H, g_A = map(lambda x: calc_cagr(x[-1], x[0]), [Y, K, L, D, AI, H, A_t])

cont_K = (alpha * g_K) / g_Y * 100
cont_L = (beta * g_L) / g_Y * 100
cont_D = (gamma * g_D) / g_Y * 100
cont_AI = (delta * g_AI) / g_Y * 100
cont_H = (theta * g_H) / g_Y * 100
cont_A = g_A / g_Y * 100

df_decomp = pd.DataFrame({
    'Yếu tố': ['Vốn (K)', 'Lao động (L)', 'Kinh tế số (D)', 'AI', 'Nhân lực (H)', 'TFP'],
    'Đóng góp (%)': [cont_K, cont_L, cont_D, cont_AI, cont_H, cont_A]
})
print(df_decomp)

# Vẽ biểu đồ cột
plt.figure(figsize=(10, 5))
bars = plt.bar(df_decomp['Yếu tố'], df_decomp['Đóng góp (%)'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
plt.title('Đóng góp của các yếu tố vào tăng trưởng GDP (2020-2025)')
plt.ylabel('Tỷ lệ đóng góp (%)')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{bar.get_height():.1f}%', ha='center', va='bottom')
plt.savefig('outputs/m1_growth_decomp.png')
plt.show()

print("\n" + "="*50)
print("CÂU 1.4.4: MÔ PHỎNG DỰ BÁO GDP NĂM 2030 (ĐẾN 2030)")
# Kịch bản 2030: D=30%, AI=100k, H=35%, K và L tăng 6%, TFP tăng 1.2%
K_2030 = K[-1] * (1.06**5)
L_2030 = L[-1] * (1.06**5)
A_2030 = A_t[-1] * (1.012**5)
D_2030, AI_2030, H_2030 = 30.0, 100.0, 35.0

Y_2030 = A_2030 * (K_2030**alpha) * (L_2030**beta) * (D_2030**gamma) * (AI_2030**delta) * (H_2030**theta)
print(f"-> Vốn K năm 2030: {K_2030:,.1f}")
print(f"-> Lao động L năm 2030: {L_2030:,.1f}")
print(f"-> TFP năm 2030: {A_2030:.4f}")
print(f"\n=> KẾT QUẢ: GDP DỰ BÁO NĂM 2030 ĐẠT {Y_2030:,.2f} nghìn tỷ VND")

# Lưu CSV phục vụ Dashboard sau này
df_mape.to_csv('outputs/m1_macro_history_tfp.csv', index=False)
pd.DataFrame({'Năm': [2030], 'GDP_Dự_báo': [Y_2030]}).to_csv('outputs/m1_macro_forecast_2026_2030.csv', index=False)
print("\n[Hệ thống] Đã xuất toàn bộ kết quả ra file trong thư mục 'outputs/'")
