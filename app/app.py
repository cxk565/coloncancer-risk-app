import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import importlib
import pkgutil

# ==========================================
# 0. Matplotlib 后台设置：必须放在 pyplot 前
# ==========================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================
# 1. TabICL 核心导入
# 注意：这里不要导入 tabicl.shap
# ==========================================
from tabicl import TabICLClassifier


# ==========================================
# 2. 页面配置与高级 CSS 美化
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
# 3. 头部设计
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
    <b>📊 System Introduction:</b> Powered by <b>TabICLv2</b>—an advanced In-Context Learning foundation model—this platform integrates key clinical indicators to dynamically predict the risk of postoperative hypoalbuminemia. It features real-time <b>SHAP (SHapley Additive exPlanations)</b> interpretations, providing clinicians with explainable decision support.
    </span>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 4. 模型加载：兼容旧 pickle + sklearn SimpleImputer
# ==========================================
def patch_old_tabicl_sklearn_modules():
    """
    解决旧版 pickle 模型中的 TabICL 路径问题：
    旧路径：tabicl.sklearn.*
    新路径：tabicl._sklearn.*
    """
    import tabicl as tabicl_pkg

    try:
        new_pkg = importlib.import_module("tabicl._sklearn")

        # 让旧的 tabicl.sklearn 指向新的 tabicl._sklearn
        sys.modules["tabicl.sklearn"] = new_pkg
        setattr(tabicl_pkg, "sklearn", new_pkg)

        # 自动映射 tabicl._sklearn 下面所有子模块
        if hasattr(new_pkg, "__path__"):
            for module_info in pkgutil.walk_packages(
                new_pkg.__path__,
                prefix="tabicl._sklearn."
            ):
                new_module_name = module_info.name
                old_module_name = new_module_name.replace(
                    "tabicl._sklearn",
                    "tabicl.sklearn",
                    1
                )

                try:
                    sys.modules[old_module_name] = importlib.import_module(new_module_name)
                except Exception:
                    pass

    except Exception:
        # 自动映射失败时，使用常见模块手动映射
        manual_map = {
            "tabicl.sklearn": "tabicl._sklearn",
            "tabicl.sklearn.preprocessing": "tabicl._sklearn.preprocessing",
            "tabicl.sklearn.classifier": "tabicl._sklearn.classifier",
            "tabicl.sklearn.regressor": "tabicl._sklearn.regressor",
            "tabicl.sklearn.sklearn_utils": "tabicl._sklearn.sklearn_utils",
            "tabicl.sklearn.utils": "tabicl._sklearn.utils",
        }

        for old_name, new_name in manual_map.items():
            try:
                sys.modules[old_name] = importlib.import_module(new_name)
            except Exception:
                pass


class TabICLCompatUnpickler(pickle.Unpickler):
    """
    兜底：
    如果 pickle 仍然要求 tabicl.sklearn.xxx，
    就在 find_class 时重定向到 tabicl._sklearn.xxx。
    """
    def find_class(self, module, name):
        if module.startswith("tabicl.sklearn"):
            new_module = module.replace("tabicl.sklearn", "tabicl._sklearn", 1)
            try:
                return super().find_class(new_module, name)
            except Exception:
                pass

        return super().find_class(module, name)


def patch_sklearn_pickle_compat(obj):
    """
    修复 sklearn 版本不一致导致的旧 pickle 问题。

    当前主要修复：
    AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'

    因为本 app 输入全部是数值型临床变量，所以补成 float64 是安全的。
    """
    visited = set()

    def walk(x):
        obj_id = id(x)

        if obj_id in visited:
            return

        visited.add(obj_id)

        # 1. 修复 SimpleImputer
        if x.__class__.__name__ == "SimpleImputer":
            if not hasattr(x, "_fill_dtype"):
                if hasattr(x, "statistics_") and hasattr(x.statistics_, "dtype"):
                    x._fill_dtype = x.statistics_.dtype
                else:
                    x._fill_dtype = np.dtype("float64")

            if not hasattr(x, "_fit_dtype"):
                if hasattr(x, "statistics_") and hasattr(x.statistics_, "dtype"):
                    x._fit_dtype = x.statistics_.dtype
                else:
                    x._fit_dtype = np.dtype("float64")

            if not hasattr(x, "keep_empty_features"):
                x.keep_empty_features = False

        # 2. 遍历 dict
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
            return

        # 3. 遍历 list / tuple / set
        if isinstance(x, (list, tuple, set)):
            for v in x:
                walk(v)
            return

        # 4. 遍历普通对象属性
        if hasattr(x, "__dict__"):
            for v in vars(x).values():
                walk(v)

    walk(obj)
    return obj


@st.cache_resource
def load_model():
    patch_old_tabicl_sklearn_modules()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "tabicl_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model = TabICLCompatUnpickler(f).load()

    # 关键修复：补 sklearn 旧 pickle 缺失属性
    model = patch_sklearn_pickle_compat(model)

    return model


try:
    model = load_model()
except Exception as e:
    st.error(
        "🚨 Model loading failed. Please ensure 'tabicl_model.pkl' is uploaded. "
        f"Error details: {e}"
    )
    st.exception(e)
    st.stop()


# ==========================================
# 5. 输入变量默认值
# ==========================================
default_values = {
    "ChE": 6664.0,
    "Age": 816.0,
    "PA": 197.5,
    "Crea": 69.7,
    "FDP": 1.5,
    "Lymph_pct": 24.2,
    "CEA": 3.57,
    "GLO": 28.6,
    "Lymph_count": 1.59
}

for key, val in default_values.items():
    if f"{key}_slider" not in st.session_state:
        st.session_state[f"{key}_slider"] = val
    if f"{key}_num" not in st.session_state:
        st.session_state[f"{key}_num"] = val


def sync_inputs(src_key, dest_key):
    st.session_state[dest_key] = st.session_state[src_key]


# ==========================================
# 6. 侧边栏输入
# ==========================================
st.sidebar.markdown("### 🖥️ System Status")
st.sidebar.success("🟢 Core Engine: TabICLv2 Ready")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Rapid Parameter Adjustment")

with st.sidebar.expander("👤 Demographics & Hepatorenal", expanded=True):
    st.slider(
        "Age (Months)",
        200.0,
        1300.0,
        step=1.0,
        key="Age_slider",
        on_change=sync_inputs,
        args=("Age_slider", "Age_num")
    )
    st.slider(
        "Creatinine (Crea) μmol/L",
        10.0,
        1200.0,
        step=0.1,
        key="Crea_slider",
        on_change=sync_inputs,
        args=("Crea_slider", "Crea_num")
    )
    st.slider(
        "Prealbumin (PA) mg/L",
        10.0,
        800.0,
        step=1.0,
        key="PA_slider",
        on_change=sync_inputs,
        args=("PA_slider", "PA_num")
    )
    st.slider(
        "Globulin (GLO) g/L",
        10.0,
        120.0,
        step=0.1,
        key="GLO_slider",
        on_change=sync_inputs,
        args=("GLO_slider", "GLO_num")
    )

with st.sidebar.expander("🩸 Hematological Indices", expanded=True):
    st.slider(
        "Lymphocyte Percentage (Lymph%)",
        0.0,
        100.0,
        step=0.1,
        key="Lymph_pct_slider",
        on_change=sync_inputs,
        args=("Lymph_pct_slider", "Lymph_pct_num")
    )
    st.slider(
        "Lymphocyte Count (×10^9/L)",
        0.0,
        50.0,
        step=0.01,
        key="Lymph_count_slider",
        on_change=sync_inputs,
        args=("Lymph_count_slider", "Lymph_count_num")
    )
    st.slider(
        "Fibrin Degradation Products (FDP) mg/L",
        0.0,
        300.0,
        step=0.01,
        key="FDP_slider",
        on_change=sync_inputs,
        args=("FDP_slider", "FDP_num")
    )

with st.sidebar.expander("🔬 Specific Enzymes & Markers", expanded=True):
    st.slider(
        "Cholinesterase (ChE) U/L",
        100.0,
        25000.0,
        step=10.0,
        key="ChE_slider",
        on_change=sync_inputs,
        args=("ChE_slider", "ChE_num")
    )
    st.slider(
        "Carcinoembryonic Antigen (CEA) ng/mL",
        0.0,
        5000.0,
        step=0.1,
        key="CEA_slider",
        on_change=sync_inputs,
        args=("CEA_slider", "CEA_num")
    )


# ==========================================
# 7. 主界面输入
# ==========================================
st.markdown("### 👨‍⚕️ Clinical Parameter Input Matrix")
st.markdown("*(Enter exact values below, or use the sidebar sliders to adjust synchronously)*")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.number_input(
        "Age (Months)",
        min_value=200.0,
        max_value=1300.0,
        step=1.0,
        format="%.0f",
        key="Age_num",
        on_change=sync_inputs,
        args=("Age_num", "Age_slider")
    )
    st.number_input(
        "Crea (μmol/L)",
        min_value=10.0,
        max_value=1200.0,
        step=0.1,
        format="%.1f",
        key="Crea_num",
        on_change=sync_inputs,
        args=("Crea_num", "Crea_slider")
    )

with col2:
    st.number_input(
        "PA (mg/L)",
        min_value=10.0,
        max_value=800.0,
        step=1.0,
        format="%.1f",
        key="PA_num",
        on_change=sync_inputs,
        args=("PA_num", "PA_slider")
    )
    st.number_input(
        "GLO (g/L)",
        min_value=10.0,
        max_value=120.0,
        step=0.1,
        format="%.1f",
        key="GLO_num",
        on_change=sync_inputs,
        args=("GLO_num", "GLO_slider")
    )

with col3:
    st.number_input(
        "Lymph (%)",
        min_value=0.0,
        max_value=100.0,
        step=0.1,
        format="%.1f",
        key="Lymph_pct_num",
        on_change=sync_inputs,
        args=("Lymph_pct_num", "Lymph_pct_slider")
    )
    st.number_input(
        "Lymph Count",
        min_value=0.0,
        max_value=50.0,
        step=0.01,
        format="%.2f",
        key="Lymph_count_num",
        on_change=sync_inputs,
        args=("Lymph_count_num", "Lymph_count_slider")
    )

with col4:
    st.number_input(
        "ChE (U/L)",
        min_value=100.0,
        max_value=25000.0,
        step=10.0,
        format="%.0f",
        key="ChE_num",
        on_change=sync_inputs,
        args=("ChE_num", "ChE_slider")
    )
    st.number_input(
        "CEA (ng/mL)",
        min_value=0.0,
        max_value=5000.0,
        step=0.1,
        format="%.2f",
        key="CEA_num",
        on_change=sync_inputs,
        args=("CEA_num", "CEA_slider")
    )

col5, col6, col7, col8 = st.columns(4)

with col5:
    st.number_input(
        "FDP (mg/L)",
        min_value=0.0,
        max_value=300.0,
        step=0.01,
        format="%.2f",
        key="FDP_num",
        on_change=sync_inputs,
        args=("FDP_num", "FDP_slider")
    )


# ==========================================
# 8. 构建输入矩阵：严格保持训练时变量顺序
# ==========================================
expected_features = [
    "ChE",
    "Age",
    "PA",
    "Crea",
    "FDP",
    "Lymph%",
    "CEA",
    "GLO",
    "Lymphocyte count"
]

input_df = pd.DataFrame({
    "ChE": [st.session_state["ChE_num"]],
    "Age": [st.session_state["Age_num"]],
    "PA": [st.session_state["PA_num"]],
    "Crea": [st.session_state["Crea_num"]],
    "FDP": [st.session_state["FDP_num"]],
    "Lymph%": [st.session_state["Lymph_pct_num"]],
    "CEA": [st.session_state["CEA_num"]],
    "GLO": [st.session_state["GLO_num"]],
    "Lymphocyte count": [st.session_state["Lymph_count_num"]]
})

input_df = input_df[expected_features]


# ==========================================
# 9. 预测与解释
# ==========================================
if st.button("🚀 Run TabICLv2 Risk Assessment", type="primary"):

    with st.spinner("🧬 In-Context Learning model is analyzing clinical features..."):

        # ------------------------------
        # 9.1 模型预测
        # ------------------------------
        try:
            risk_prob = model.predict_proba(input_df)[0][1]
            risk_prob = float(risk_prob)

        except Exception as e:
            st.error("模型预测失败。当前变量顺序如下。若变量没问题，则多半是 sklearn pickle 版本兼容问题。")
            st.write("Current input columns:", list(input_df.columns))
            st.exception(e)
            st.stop()

        # ------------------------------
        # 9.2 显示预测结果
        # ------------------------------
        st.markdown("---")
        st.markdown("### 🎯 Postoperative Risk Inference Report")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.metric(
                label="Probability of Hypoalbuminemia",
                value=f"{risk_prob * 100:.2f} %"
            )

        with res_col2:
            st.markdown("<br>", unsafe_allow_html=True)

            if risk_prob > 0.5:
                st.error(
                    "🚨 **[HIGH RISK ALERT]** The model identifies this patient as highly susceptible to "
                    "**postoperative hypoalbuminemia**. Intensive perioperative nutritional management and "
                    "enhanced postoperative monitoring are strongly recommended."
                )
                st.toast("High-risk alert detected!", icon="⚠️")
            else:
                st.success(
                    "✅ **[SAFE ASSESSMENT]** The patient is currently in the low-risk zone. "
                    "Maintenance of standard postoperative care protocols is recommended."
                )
                st.balloons()

        # ------------------------------
        # 9.3 SHAP 解释
        # ------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🧠 Risk Factor Attribution")
        st.info(
            "💡 **Interpretation Guide:** Red indicates risk-increasing factors, "
            "while blue indicates protective factors."
        )

        try:
            # 关键：延迟导入，防止 app 启动时因 SHAP 失败而崩溃
            import shap
            from tabicl.shap import get_shap_values

            shap_vals_raw = get_shap_values(
                estimator=model,
                X_test=input_df,
                attribute_names=expected_features
            )

            vals_matrix = shap_vals_raw.values if hasattr(shap_vals_raw, "values") else shap_vals_raw
            vals_matrix = np.asarray(vals_matrix)

            if vals_matrix.ndim == 3:
                shap_val_single = vals_matrix[0, :, 1]
            elif vals_matrix.ndim == 2:
                shap_val_single = vals_matrix[0]
            else:
                shap_val_single = vals_matrix

            shap_val_single = np.asarray(shap_val_single, dtype=float)

            base_val = float(risk_prob - np.sum(shap_val_single))

            exp = shap.Explanation(
                values=shap_val_single,
                base_values=base_val,
                data=input_df.iloc[0].values,
                feature_names=expected_features
            )

            tab1, tab2, tab3, tab4 = st.tabs([
                "🌊 Waterfall Plot",
                "⚖️ Force Plot",
                "📈 Decision Plot",
                "📊 Bar Plot"
            ])

            with tab1:
                st.markdown("#### 1. Local Waterfall Plot")
                fig = plt.figure(figsize=(10, 6))
                shap.waterfall_plot(exp, max_display=10, show=False)
                st.pyplot(fig, bbox_inches="tight")
                plt.close(fig)

            with tab2:
                st.markdown("#### 2. Local Force Plot")
                shap.force_plot(
                    base_val,
                    shap_val_single,
                    input_df.iloc[0],
                    matplotlib=True,
                    show=False
                )
                fig = plt.gcf()
                st.pyplot(fig, bbox_inches="tight")
                plt.close(fig)

            with tab3:
                st.markdown("#### 3. Decision Plot")
                shap.decision_plot(
                    base_val,
                    shap_val_single,
                    input_df.iloc[0].values,
                    feature_names=expected_features,
                    show=False
                )
                fig = plt.gcf()
                st.pyplot(fig, bbox_inches="tight")
                plt.close(fig)

            with tab4:
                st.markdown("#### 4. Absolute Impact Bar Plot")
                fig = plt.figure(figsize=(10, 6))
                shap.plots.bar(exp, max_display=10, show=False)
                st.pyplot(fig, bbox_inches="tight")
                plt.close(fig)

        except ImportError as e:
            st.warning(
                "SHAP解释模块暂时不可用，但模型预测功能已经完成。"
                "请检查 requirements.txt 是否包含 tabicl[shap]、shap、shapiq、numba 和 matplotlib。"
            )
            st.exception(e)

        except Exception as e:
            st.error(f"An error occurred while generating the SHAP plots: {e}")
            st.exception(e)
