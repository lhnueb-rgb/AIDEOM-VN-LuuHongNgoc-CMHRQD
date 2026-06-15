import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# ĐỒ ÁN TỔNG HỢP AIDEOM-VN (BÀI 12)
# ==============================================================================

st.set_page_config(page_title="AIDEOM-VN Dashboard", layout="wide", page_icon="🚀")

st.title("🚀 Hệ thống hỗ trợ ra quyết định AIDEOM-VN (2026-2030)")
st.markdown("**Đồ án:** Mô hình ra quyết định phát triển kinh tế | **Tác giả:** Lưu Hồng Ngọc")

# --- DATA MÔ PHỎNG (Lấy từ kết quả M1-M5) ---
# 5 Kịch bản chính sách (Tỷ trọng: K, D, AI, H)
scenarios = {
    "S1. Truyền thống": [70, 10, 10, 10],
    "S2. Số hóa nhanh": [25, 45, 15, 15],
    "S3. AI dẫn dắt": [20, 20, 45, 15],
    "S4. Bao trùm số": [30, 20, 10, 40],
    "S5. Tối ưu cân bằng (AIDEOM-VN)": [35, 25, 20, 20]
}

# --- GIAO DIỆN SIDEBAR ---
st.sidebar.header("⚙️ Tùy chỉnh Chính sách")
selected_scenario = st.sidebar.radio("Chọn kịch bản phân bổ (80.000 tỷ VND):", list(scenarios.keys()))
budget = 80000

# Trích xuất tỷ trọng
alloc = scenarios[selected_scenario]
alloc_dict = {"Vốn vật chất (K)": alloc[0], "Hạ tầng số (D)": alloc[1], "Trí tuệ nhân tạo (AI)": alloc[2], "Nhân lực số (H)": alloc[3]}

# --- TẠO 4 TAB NỘI DUNG CHÍNH ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Tổng quan Vĩ mô (M1&M2)", 
    "💰 2. Phân bổ Ngân sách (M3)", 
    "⚖️ 3. Kịch bản So sánh (M4)", 
    "⚠️ 4. Cảnh báo Rủi ro (M5)"
])

# ==========================================
# TAB 1: TỔNG QUAN VĨ MÔ
# ==========================================
with tab1:
    st.subheader("Dự báo Kinh tế Vĩ mô & Sẵn sàng số")
    col1, col2, col3 = st.columns(3)
    col1.metric("Dự phóng GDP 2030 (Tỷ USD)", "685.4", "+8.2%")
    col2.metric("Tỷ trọng Kinh tế số", "30.5%", "+11% so với 2025")
    col3.metric("Xếp hạng AI Readiness", "Top 4 ASEAN", "Tăng 2 bậc")
    
    st.markdown("##### Bản đồ năng lực 6 Vùng kinh tế (M2)")
    # Dữ liệu radar chart cho 6 vùng
    categories = ['Tăng trưởng', 'FDI', 'Digital Index', 'AI Readiness', 'Nhân lực']
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=[8, 9, 7, 6, 8], theta=categories, fill='toself', name='Đồng bằng sông Hồng'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=[9, 8, 8, 7, 9], theta=categories, fill='toself', name='Đông Nam Bộ'
    ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)

# ==========================================
# TAB 2: PHÂN BỔ NGÂN SÁCH
# ==========================================
with tab2:
    st.subheader(f"Cơ cấu Phân bổ Ngân sách: {selected_scenario}")
    
    df_alloc = pd.DataFrame({
        "Hạng mục": list(alloc_dict.keys()),
        "Tỷ lệ (%)": list(alloc_dict.values()),
        "Ngân sách (Tỷ VND)": [(x/100)*budget for x in alloc_dict.values()]
    })
    
    colA, colB = st.columns([1, 1])
    with colA:
        fig_pie = px.pie(df_alloc, values='Tỷ lệ (%)', names='Hạng mục', hole=0.4, 
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie, use_container_width=True)
    with colB:
        st.dataframe(df_alloc, use_container_width=True)
        st.info("Kịch bản này phản ánh các định hướng tại Mục 15 của bài báo nguồn, phân bổ dựa trên trọng số chiến lược quốc gia.")

# ==========================================
# TAB 3: KỊCH BẢN SO SÁNH (LAO ĐỘNG & GDP)
# ==========================================
with tab3:
    st.subheader("Mô phỏng NetJob và GDP theo Kịch bản")
    
    # Giả lập dữ liệu kết quả từ M4
    df_compare = pd.DataFrame({
        "Kịch bản": ["S1", "S2", "S3", "S4", "S5"],
        "GDP Tăng thêm (Tỷ VND)": [12000, 16500, 18200, 14000, 17500],
        "NetJob (Việc làm)": [150000, 320000, 450000, 600000, 520000]
    })
    
    fig_bar = px.bar(df_compare, x="Kịch bản", y="GDP Tăng thêm (Tỷ VND)", text_auto=True,
                     title="So sánh Tăng trưởng GDP giữa 5 Kịch bản",
                     color="GDP Tăng thêm (Tỷ VND)", color_continuous_scale="Blues")
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("Nhận xét: Kịch bản **S5 (Tối ưu cân bằng)** mang lại mức tăng trưởng GDP sát với S3 nhưng bảo đảm lượng việc làm (NetJob) sinh ra vượt trội nhờ phân bổ đồng đều vào Nhân lực (H).")

# ==========================================
# TAB 4: CẢNH BÁO RỦI RO
# ==========================================
with tab4:
    st.subheader("Đánh giá Đánh đổi Đa mục tiêu (M5)")
    st.warning("Mô phỏng dựa trên tối ưu Pareto (Bài 7) và Ngẫu nhiên 2 giai đoạn (Bài 10).")
    
    df_risk = pd.DataFrame({
        "Mục tiêu": ["Tăng trưởng", "Bao trùm", "Phát thải", "An ninh mạng"],
        "Giá trị": [0.85, 0.75, 0.40, 0.30]  # Thang chuẩn hóa 0-1
    })
    
    fig_risk = px.line_polar(df_risk, r='Giá trị', theta='Mục tiêu', line_close=True,
                             title="Đa giác Radar: Rủi ro vs Lợi ích (Kịch bản S5)")
    fig_risk.update_traces(fill='toself', line_color='red')
    st.plotly_chart(fig_risk, use_container_width=True)
    
    st.markdown("""
    **Khuyến nghị Chính sách:**
    * Nếu chỉ chạy theo S3 (AI dẫn dắt), rủi ro an ninh mạng và sa thải lao động (chỉ số Bao trùm) sẽ chạm ngưỡng nguy hiểm.
    * Kịch bản S5 duy trì một khoản dự phòng 15.000 tỷ VND (wait-and-see) giúp thích ứng tốt nếu xảy ra cú sốc tương tự đại dịch hoặc thiên tai.
    """)
