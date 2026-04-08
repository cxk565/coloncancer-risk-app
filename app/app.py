import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os

# 强制后台绘图防止多线程崩溃
import matplotlib
matplotlib.use('Agg')

# ✨ 引入 TabICLv2 核心与 SHAP 模块
from tabicl import TabICLClassifier
from tabicl.shap import get_shap_values

# ==========================================
# 0. 页面配置与高级 CSS 美化
# ==========================================
st.set_page_config(
    page_title="Hypoalbuminemia Risk Predictor (TabICLv2)",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded" 
)

st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Times New Roman', sans-serif;
    }
    div.stButton > button:first-child {
        background-color: #2E86C1;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 15px;
    }
    div.stButton > button:first-child:hover {
        background-color: #1B4F72;
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    [data-testid="stSidebar"] {
        background-color: #F8F9F9;
        border-right: 1px solid #E5E7E9;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.8rem;
        color: #C0392B; 
        font-weight: 900;
    }
    input[type="number"] {
        font-weight: bold;
        color: #154360;
        background-color: #F4F6F7;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 头部设计
# ==========================================
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80) 
with col_title:
    st.title("Intelligent Warning Platform for Postoperative Hypoalbuminemia Risk in Colon Cancer")
    st.markdown("**(Powered by TabICLv2: A State-of-the-Art Tabular Foundation Model)**")

st.markdown("""
<div style='background-color: #EBF5FB; padding: 15px; border-radius: 10px; border-left: 5px solid #2980B9; margin-bottom: 25px;'>
    <span style='color: #154360; font-size: 15px;'>
    <b>📊 System Introduction:</b> Powered by <b>TabICLv2</b>—an advanced In-Context Learning foundation model—this platform integrates key clinical indicators to dynamically predict the risk of postoperative hypoalbuminemia. It features real-time <b>SHAP (SHapley Additive exPlanations)</b> interpretations, providing clinicians with unprecedented accuracy and explainable decision support.
    </span>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心引擎加载
# ==========================================
@st.cache_resource 
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "tabicl_model.pkl") 
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"🚨 Model loading failed. Please ensure 'tabicl_model.pkl' is uploaded. Error details: {e}")
    st.stop()

# ==========================================
# 3. 侧边栏与主界面双向绑定
# ==========================================
default_values = {
    'ChE': 6664.0, 'Age': 816.0, 'PA': 197.5, 
    'Crea': 69.7, 'FDP': 1.5, 'Lymph_pct': 24.2, 
    'CEA': 3.57, 'GLO': 28.6, 'Lymph_count': 1.59
}

for key, val in default_values.items():
    if f"{key}_slider" not in st.session_state:
        st.session_state[f"{key}_slider"] = val
    if f"{key}_num" not in st.session_state:
        st.session_state[f"{key}_num"] = val

def sync_inputs(src_key, dest_key):
    st.session_state[dest_key] = st.session_state[src_key]

st.sidebar.markdown("### 🖥️ System Status")
st.sidebar.success("🟢 Core Engine: TabICLv2 Ready")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Rapid Parameter Adjustment")

with st.sidebar.expander("👤 Demographics & Hepatorenal", expanded=True):
    st.slider("Age (Months)", 200.0, 1300.0, step=1.0, key="Age_slider", on_change=sync_inputs, args=("Age_slider", "Age_num"))
    st.slider("Creatinine (Crea) μmol/L", 10.0, 1200.0, step=0.1, key="Crea_slider", on_change=sync_inputs, args=("Crea_slider", "Crea_num"))
    st.slider("Prealbumin (PA) mg/L", 10.0, 800.0, step=1.0, key="PA_slider", on_change=sync_inputs, args=("PA_slider", "PA_num"))
    st.slider("Globulin (GLO) g/L", 10.0, 120.0, step=0.1, key="GLO_slider", on_change=sync_inputs, args=("GLO_slider", "GLO_num"))

with st.sidebar.expander("🩸 Hematological Indices", expanded=True):
    st.slider("Lymphocyte Percentage (Lymph%)", 0.0, 100.0, step=0.1, key="Lymph_pct_slider", on_change=sync_inputs, args=("Lymph_pct_slider", "Lymph_pct_num"))
    st.slider("Lymphocyte Count (×10^9/L)", 0.0, 50.0, step=0.01, key="Lymph_count_slider", on_change=sync_inputs, args=("Lymph_count_slider", "Lymph_count_num"))
    st.slider("Fibrin Degradation Products (FDP) mg/L", 0.0, 300.0, step=0.01, key="FDP_slider", on_change=sync_inputs, args=("FDP_slider", "FDP_num"))
    
with st.sidebar.expander("🔬 Specific Enzymes & Markers", expanded=True):
    st.slider("Cholinesterase (ChE) U/L", 100.0, 25000.0, step=10.0, key="ChE_slider", on_change=sync_inputs, args=("ChE_slider", "ChE_num"))
    st.slider("Carcinoembryonic Antigen (CEA) ng/mL", 0.0, 5000.0, step=0.1, key="CEA_slider", on_change=sync_inputs, args=("CEA_slider", "CEA_num"))

st.markdown("### 👨‍⚕️ Clinical Parameter Input Matrix")
st.markdown("*(Enter exact values below, or use the sidebar sliders to adjust synchronously)*")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.number_input("Age (Months)", min_value=200.0, max_value=1300.0, step=1.0, format="%.0f", key="Age_num", on_change=sync_inputs, args=("Age_num", "Age_slider"))
    st.number_input("Crea (μmol/L)", min_value=10.0, max_value=1200.0, step=0.1, format="%.1f", key="Crea_num", on_change=sync_inputs, args=("Crea_num", "Crea_slider"))
with col2:
    st.number_input("PA (mg/L)", min_value=10.0, max_value=800.0, step=1.0, format="%.1f", key="PA_num", on_change=sync_inputs, args=("PA_num", "PA_slider"))
    st.number_input("GLO (g/L)", min_value=10.0, max_value=120.0, step=0.1, format="%.1f", key="GLO_num", on_change=sync_inputs, args=("GLO_num", "GLO_slider"))
with col3:
    st.number_input("Lymph (%)", min_value=0.0, max_value=100.0, step=0.1, format="%.1f", key="Lymph_pct_num", on_change=sync_inputs, args=("Lymph_pct_num", "Lymph_pct_slider"))
    st.number_input("Lymph Count", min_value=0.0, max_value=50.0, step=0.01, format="%.2f", key="Lymph_count_num", on_change=sync_inputs, args=("Lymph_count_num", "Lymph_count_slider"))
with col4:
    st.number_input("ChE (U/L)", min_value=100.0, max_value=25000.0, step=10.0, format="%.0f", key="ChE_num", on_change=sync_inputs, args=("ChE_num", "ChE_slider"))
    st.number_input("CEA (ng/mL)", min_value=0.0, max_value=5000.0, step=0.1, format="%.2f", key="CEA_num", on_change=sync_inputs, args=("CEA_num", "CEA_slider"))

col5, col6, col7, col8 = st.columns(4)
with col5:
    st.number_input("FDP (mg/L)", min_value=0.0, max_value=300.0, step=0.01, format="%.2f", key="FDP_num", on_change=sync_inputs, args=("FDP_num", "FDP_slider"))


# ✨ 核心修复：根据 CSV 截图严格排列表头顺序
expected_features = ['ChE', 'Age', 'PA', 'Crea', 'FDP', 'Lymph%', 'CEA', 'GLO', 'Lymphocyte count']

# 按照截图顺序将 session_state 填入
input_df = pd.DataFrame({
    'ChE': [st.session_state["ChE_num"]], 
    'Age': [st.session_state["Age_num"]], 
    'PA': [st.session_state["PA_num"]], 
    'Crea': [st.session_state["Crea_num"]],
    'FDP': [st.session_state["FDP_num"]], 
    'Lymph%': [st.session_state["Lymph_pct_num"]], 
    'CEA': [st.session_state["CEA_num"]], 
    'GLO': [st.session_state["GLO_num"]],
    'Lymphocyte count': [st.session_state["Lymph_count_num"]] 
})

# 强制重排顺序，确保绝对一致
input_df = input_df[expected_features]

# ==========================================
# 4. TabICLv2 前向推理与 SHAP 动态解析
# ==========================================
if st.button("🚀 Run TabICLv2 Risk Assessment", type="primary"):
    with st.spinner('🧬 In-Context Learning model is analyzing clinical features...'):
        
        risk_prob = model.predict_proba(input_df)[0][1] 
        
        st.markdown("---")
        st.markdown("### 🎯 Postoperative Risk Inference Report")
        
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric(label="Probability of Hypoalbuminemia", value=f"{risk_prob * 100:.2f} %")
            
        with res_col2:
            st.markdown("<br>", unsafe_allow_html=True) 
            if risk_prob > 0.5: 
                st.error("🚨 **[HIGH RISK ALERT]** The model identifies this patient as highly susceptible to **postoperative hypoalbuminemia**. Intensive perioperative nutritional management and enhanced postoperative monitoring are strongly recommended.")
                st.toast('High-risk alert detected!', icon='⚠️') 
            else:
                st.success("✅ **[SAFE ASSESSMENT]** The patient is currently in the low-risk zone. Maintenance of standard postoperative care protocols is recommended.")
                st.balloons() 

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🧠 Risk Factor Attribution (TabICLv2 Native SHAP)")
        st.info("💡 **Interpretation Guide:** Explore different tabs to view the explanations. Red color indicates risk-increasing factors, while blue indicates protective factors.")
        
        try:
            # ✨ 调用 TabICLv2 官方原生高速 SHAP 解析器
            shap_vals_raw = get_shap_values(
                estimator=model,
                X_test=input_df,
                attribute_names=expected_features
            )
            
            # ✨ 终极修复点1：彻底剥离 Explanation 外壳
            vals_matrix = shap_vals_raw.values if hasattr(shap_vals_raw, 'values') else shap_vals_raw
            
            # ✨ 终极修复点2：精准维度切片提取正类 1D 数组
            if len(vals_matrix.shape) == 3:
                shap_val_single = vals_matrix[0, :, 1]
            elif len(vals_matrix.shape) == 2:
                shap_val_single = vals_matrix[0]
            else:
                shap_val_single = vals_matrix
            
            # ✨ 神级Hack：此时的 shap_val_single 必定是 1D 纯数字矩阵，np.sum 完美执行
            base_val = risk_prob - np.sum(shap_val_single)
            
            # 用纯数字干净、安全地重新组装单一解释的 Explanation 对象
            exp = shap.Explanation(values=shap_val_single, base_values=base_val, 
                                   data=input_df.iloc[0], feature_names=expected_features)
            
            tab1, tab2, tab3, tab4 = st.tabs(["🌊 Waterfall Plot", "⚖️ Force Plot", "📈 Decision Plot", "📊 Bar Plot"])
            
            with tab1:
                st.markdown("#### 1. Local Waterfall Plot")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(exp, max_display=10, show=False)
                st.pyplot(fig, bbox_inches='tight')
                plt.close(fig)
                
            with tab2:
                st.markdown("#### 2. Local Force Plot")
                shap.force_plot(base_val, shap_val_single, input_df.iloc[0], matplotlib=True, show=False)
                st.pyplot(plt.gcf(), bbox_inches='tight')
                plt.clf()
                
            with tab3:
                st.markdown("#### 3. Decision Plot")
                shap.decision_plot(base_val, shap_val_single, input_df.iloc[0], show=False)
                st.pyplot(plt.gcf(), bbox_inches='tight')
                plt.clf()
                
            with tab4:
                st.markdown("#### 4. Absolute Impact Bar Plot")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.bar(exp, max_display=10, show=False)
                st.pyplot(fig, bbox_inches='tight')
                plt.close(fig)
            
        except Exception as e:
            st.error(f"An error occurred while generating the SHAP plots: {e}")
