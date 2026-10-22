import streamlit as st
import pandas as pd
import numpy as np
import pydicom
import cv2
import torch
import joblib
import json
import os
import sys
from pathlib import Path
import SimpleITK as sitk
from radiomics import featureextractor
import plotly.graph_objects as go
from sklearn.base import BaseEstimator, TransformerMixin

from unet_model import SimpleUNet
import io

# ==========================================
# 0. 基础配置与多语言字典
# ==========================================
torch.backends.cudnn.enabled = False

st.set_page_config(page_title="RMPP Classification Support", page_icon="RMPP", layout="wide")

# 多语言配置字典
LANG = {
    "zh": {
        "title": "儿童难治性肺炎支原体肺炎(RMPP)影像组学分类辅助系统",
        "step1": "📥 第 1 步：上传胸部 X 线片 (支持 .dcm 格式)",
        "demo_hint": "💡 快速体验（内置脱敏示例胸片）：",
        "step2": "🧑‍⚕️ 第 2 步：输入患儿年龄及实验室指标",
        "demo_close": "✖ 关闭示例",
        "age_years_label": "患儿年龄 (岁)",
        "age_months_label": "患儿年龄 (月)",
        "crp_label": "CRP (mg/L)",
        "wbc_label": "WBC (×10⁹/L)",
        "ldh_label": "LDH (U/L)",
        "step3": "🚀 第 3 步：开始分类分析",
        "btn_run": "运行全流程分类",
        "init_sys": "正在初始化系统资产...",
        "processing": "正在处理图像并提取组学特征...",
        "col_std": "原始影像标准化",
        "col_seg": "全自动肺部分割",
        "cap_std": "标准化胸片 (1024x1024)",
        "cap_seg": "全自动分割结果 (绿色标注区域)",
        "res_title": "分类分析结果",
        "tab_svm": "SVM 独立分类",
        "tab_integrated": "多模型参考结果",
        "prob_label": "RMPP 分类概率",
        "x_axis": "风险概率",
        "avg_risk": "训练 CV AUC 加权结果",
        "high_risk": "模型分类倾向 RMPP，请结合临床和影像资料进一步评估。",
        "low_risk": "模型分类倾向非 RMPP，仍应结合临床病程进行随访判断。",
        "info": "请按步骤上传文件和输入信息。本工具为研究原型，不用于独立诊断或自动治疗决策。",
        "research_notice": "研究原型：输出为回顾性分类模型结果，仅用于辅助评估。",
        "model_info": "模型与特征信息",
        "final_features": "最终模型特征",
        "pipeline_info": "所有缺失值插补、Z-score 标准化和特征选择均封装在已保存的完整 Pipeline 中。",
        "loading_error": "模型或分割资源加载失败"
    },
    "en": {
        "title": "Pediatric RMPP Chest Radiography Classification Support System",
        "step1": "📥 Step 1: Upload Chest X-ray (.dcm)",
        "demo_hint": "💡 Quick demo (built-in de-identified chest X-ray, no upload needed):",
        "step2": "🧑‍⚕️ Step 2: Enter age & lab values",
        "demo_close": "✖ Close demo",
        "age_years_label": "Patient Age (Years)",
        "age_months_label": "Patient Age (Months)",
        "crp_label": "CRP (mg/L)",
        "wbc_label": "WBC (×10⁹/L)",
        "ldh_label": "LDH (U/L)",
        "step3": "🚀 Step 3: Start Classification",
        "btn_run": "Run Classification Pipeline",
        "init_sys": "Initializing system assets...",
        "processing": "Processing image & extracting radiomics...",
        "col_std": "Standardized Image",
        "col_seg": "Auto Lung Segmentation",
        "cap_std": "Standardized X-ray (1024x1024)",
        "cap_seg": "Auto-segmentation (Green area)",
        "res_title": "Classification Results",
        "tab_svm": "SVM Classification",
        "tab_integrated": "Multi-model Reference",
        "prob_label": "RMPP Probability",
        "x_axis": "Probability of Risk",
        "avg_risk": "Training-CV-AUC Weighted Result",
        "high_risk": "The model favors RMPP. Further clinical and imaging assessment is advised.",
        "low_risk": "The model favors non-RMPP. Clinical follow-up is still required.",
        "info": "Upload a file and enter the required information. This is a research prototype and is not intended for autonomous diagnosis or treatment decisions.",
        "research_notice": "Research prototype: outputs are retrospective classification results for decision support only.",
        "model_info": "Model and Feature Information",
        "final_features": "Final model features",
        "pipeline_info": "Missing-value imputation, Z-score standardization, and feature selection are encapsulated in the saved model pipelines.",
        "loading_error": "Failed to load model or segmentation assets"
    }
}

if 'lang' not in st.session_state:
    st.session_state['lang'] = 'zh'

col_title, col_lang = st.columns([7, 3])
with col_lang:
    lang_choice = st.radio("Language", options=['中文', 'English'], horizontal=True, label_visibility="collapsed")
    st.session_state['lang'] = 'zh' if lang_choice == '中文' else 'en'

t = LANG[st.session_state['lang']]

with col_title:
    st.title(t["title"])
st.caption(t["research_notice"])
st.divider()

# ==========================================
# 1. 资源加载
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "config"

UNET_MODEL_PATH = MODELS_DIR / "cxr_unet_best.pth"
RADIOMICS_CONFIG_PATH = CONFIG_DIR / "radiomics_config.yaml"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"

DEMO_DIR = BASE_DIR / "demo"
DEMO_FILES = sorted(DEMO_DIR.glob("*.dcm")) if DEMO_DIR.is_dir() else []
DEMO_CLIN_SUGGEST = {
    "rmpp_case_positive.dcm": ("Sample ②: RMPP Positive", 4, 11, 9.28, 11.15, 391.0),
    "rmpp_case_negative.dcm": ("Sample ①: RMPP Negative", 3, 5, 0.29, 7.85, 299.0),
}


@st.cache_resource
def get_segmentation_model():
    model = SimpleUNet(n_channels=1, n_classes=1)
    model.load_state_dict(torch.load(str(UNET_MODEL_PATH), map_location='cpu'))
    model.eval()
    return model


def load_model_config():
    if MODEL_CONFIG_PATH.exists():
        with open(str(MODEL_CONFIG_PATH), "r", encoding="utf-8") as f:
            return json.load(f)
    return None


class StrictPreprocessSelector(BaseEstimator, TransformerMixin):
    """Runtime view of the selector embedded in the strict-CV model pipelines.

    The fitted selector already contains all imputation, scaling, filtering, and
    feature-selection parameters. The app only needs transform().
    """

    def __init__(
        self,
        k=8,
        p_threshold=0.05,
        protected_features=None,
        missing_threshold=0.30,
        variance_threshold=0.0,
        random_state=42,
    ):
        self.k = k
        self.p_threshold = p_threshold
        self.protected_features = protected_features
        self.missing_threshold = missing_threshold
        self.variance_threshold = variance_threshold
        self.random_state = random_state

    def _as_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X, columns=self.feature_names_in_)

    def transform(self, X):
        frame = self._as_frame(X)
        imputed = frame[self.missing_keep_].fillna(self.impute_means_)
        imputed = imputed[self.variance_keep_]
        scaled = self.scaler_.transform(imputed)
        indices = [self.variance_keep_.index(f) for f in self.selected_features_]
        return scaled[:, indices]


# The strict-CV model pickles reference the selector as __main__.StrictPreprocessSelector.
setattr(sys.modules["__main__"], "StrictPreprocessSelector", StrictPreprocessSelector)


@st.cache_resource
def get_ml_assets():
    cfg = load_model_config()
    if not cfg:
        raise FileNotFoundError(f"Missing model configuration: {MODEL_CONFIG_PATH}")
    if cfg.get("pipeline_type") != "strict_cv_internal":
        raise ValueError(
            "This app requires a model_config.json produced by the strict "
            "cross-validation pipeline."
        )

    model_names = cfg.get("model_names", [])
    if not model_names:
        raise ValueError("No model names were found in model_config.json")

    models = {}
    for name in model_names:
        model_path = MODELS_DIR / f"model_{name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")
        models[name] = joblib.load(str(model_path))

    return models, cfg


# ==========================================
# 2. 图像处理函数
# ==========================================
def apply_standardization(dcm_path):
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array.astype(np.float32)
    if 'WindowCenter' in ds and 'WindowWidth' in ds:
        wc = ds.WindowCenter[0] if isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else ds.WindowCenter
        ww = ds.WindowWidth[0] if isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else ds.WindowWidth
        img = np.clip(img, float(wc) - float(ww) / 2, float(wc) + float(ww) / 2)
        img = (img - (float(wc) - float(ww) / 2)) / float(ww) * 255.0
    else:
        img = np.clip(img, np.percentile(img, 1), np.percentile(img, 99))
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
    if np.mean([img[0:10, 0:10], img[0:10, -10:], img[-10:, 0:10], img[-10:, -10:]]) > 128:
        img = 255.0 - img
    target_size = (1024, 1024)
    old_h, old_w = img.shape[:2]
    scale = min(target_size[0] / old_w, target_size[1] / old_h)
    new_w, new_h = int(old_w * scale), int(old_h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros(target_size, dtype=np.uint8)
    top, left = (target_size[1] - new_h) // 2, (target_size[0] - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized.astype(np.uint8)
    return canvas


def run_segmentation(img_array, model):
    img_input = cv2.resize(img_array, (512, 512)) / 255.0
    img_tensor = torch.from_numpy(img_input).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        mask = (torch.sigmoid(output) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
    return cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)


def create_visual_overlay(img, mask):
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    mask_color = np.zeros_like(img_color)
    mask_color[mask == 1] = [0, 255, 0]
    overlay = cv2.addWeighted(img_color, 0.7, mask_color, 0.3, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return overlay


# ==========================================
# 3. 主交互逻辑
# ==========================================
st.subheader(t["step1"])
uploaded_file = st.file_uploader(
    t["step1"], type=['dcm'], label_visibility="collapsed"
)

# Demo
class _DemoFile(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data)
        self.name = name

if uploaded_file is not None:
    st.session_state["demo_active"] = False

if "demo_active" not in st.session_state:
    st.session_state["demo_active"] = False

if uploaded_file is None and DEMO_FILES:
    st.markdown(f"**{t['demo_hint']}**")
    btns = st.columns(len(DEMO_FILES))
    for col, f in zip(btns, DEMO_FILES):
        meta = DEMO_CLIN_SUGGEST.get(f.name)
        btn_label = meta[0] if meta else f.stem
        with col:
            if st.button(btn_label):
                st.session_state["demo_active"] = True
                st.session_state["demo_bytes"] = f.read_bytes()
                st.session_state["demo_name"] = f.name
                st.session_state["demo_auto_run"] = True
                if meta:
                    st.session_state["demo_age_y"] = meta[1]
                    st.session_state["demo_age_m"] = meta[2]
                    st.session_state["demo_crp"] = meta[3]
                    st.session_state["demo_wbc"] = meta[4]
                    st.session_state["demo_ldh"] = meta[5]

if uploaded_file is None and st.session_state.get("demo_active"):
    if st.button(t["demo_close"]):
        st.session_state["demo_active"] = False
        st.session_state.pop("demo_bytes", None)
    else:
        b = st.session_state.get("demo_bytes")
        if b:
            uploaded_file = _DemoFile(b, st.session_state.get("demo_name", "demo.dcm"))

if uploaded_file:
    st.subheader(t["step2"])
    col_age_y, col_age_m = st.columns(2)
    with col_age_y:
        age_years = st.number_input(t["age_years_label"], min_value=0, max_value=18,
                                    value=int(st.session_state.get("demo_age_y", 6)), step=1)
    with col_age_m:
        age_months = st.number_input(t["age_months_label"], min_value=0, max_value=11,
                                     value=int(st.session_state.get("demo_age_m", 0)), step=1)
    # 自动换算为岁
    age = float(age_years) + float(age_months) / 12.0
    crp = st.number_input(t["crp_label"], min_value=0.0, max_value=500.0,
                          value=float(st.session_state.get("demo_crp", 10.0)), step=0.1)
    wbc = st.number_input(t["wbc_label"], min_value=0.0, max_value=100.0,
                          value=float(st.session_state.get("demo_wbc", 8.0)), step=0.1)
    ldh = st.number_input(t["ldh_label"], min_value=0.0, max_value=10000.0,
                          value=float(st.session_state.get("demo_ldh", 300.0)), step=1.0)

    st.subheader(t["step3"])
    if st.button(t["btn_run"], type="primary") or st.session_state.pop("demo_auto_run", False):
        try:
            with st.spinner(t["init_sys"]):
                seg_model = get_segmentation_model()
                ml_models, model_cfg = get_ml_assets()
        except Exception as exc:
            st.error(f"{t['loading_error']}: {exc}")
            st.stop()

        # 图像处理
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{t['col_std']}**")
            std_img = apply_standardization(uploaded_file)
            st.image(std_img, caption=t["cap_std"], use_column_width=True)
        with col2:
            st.markdown(f"**{t['col_seg']}**")
            mask = run_segmentation(std_img, seg_model)
            overlay_view = create_visual_overlay(std_img, mask)
            st.image(overlay_view, caption=t["cap_seg"], use_column_width=True)

        # 特征提取与预测
        st.divider()
        st.subheader(t["res_title"])

        with st.spinner(t["processing"]):
            # 组学提取
            sitk_img = sitk.GetImageFromArray(std_img)
            sitk_mask = sitk.GetImageFromArray(mask)
            extractor = featureextractor.RadiomicsFeatureExtractor(str(RADIOMICS_CONFIG_PATH))
            features = extractor.execute(sitk_img, sitk_mask)

            # 数据对齐与标准化
            raw_feature_dict = {k.replace('original_', ''): float(v) for k, v in features.items() if
                                'diagnostics' not in k}
            clinical_inputs = {"age": age, "CRP": crp, "WBC": wbc, "LDH": ldh}
            feature_order = model_cfg["feature_order"]
            full_row = {
                col: clinical_inputs.get(col, raw_feature_dict.get(col, np.nan))
                for col in feature_order
            }
            input_df = pd.DataFrame([full_row], columns=feature_order).astype(float)
            missing_count = int(input_df.isna().sum(axis=1).iloc[0])

            # Every strict-CV model is a complete Pipeline. It handles imputation,
            # Z-score standardization, and selection of the fitted feature subset.
            results = {
                name: float(model.predict_proba(input_df)[0, 1])
                for name, model in ml_models.items()
            }
            auc_weights = (
                model_cfg.get("selection_cv_auc")
                or model_cfg.get("internal_valid_auc")
                or model_cfg.get("auc_weights")
                or {}
            )

            # CV-AUC weighted analysis. Models without a configured weight receive
            # equal weight so the interface remains usable with incomplete configs.
            if not auc_weights:
                auc_weights = {name: 1.0 for name in results}
            total_weight = sum(auc_weights.get(name, 0.0) for name in results)
            weighted_avg_risk = sum(
                results[name] * (auc_weights.get(name, 0.0) / total_weight)
                for name in results
            )

            # 确定最佳模型
            best_model_name = model_cfg.get("best_model")
            if not best_model_name:
                best_model_name = max(results, key=results.get)

        # ==========================================
        # 结果展示
        # ==========================================
        tab1_label = t["tab_svm"].replace("SVM", best_model_name)
        tab1, tab2 = st.tabs([tab1_label, t["tab_integrated"]])

        # ---------- 最佳模型独立预测----------
        with tab1:
            best_prob = results.get(best_model_name, results.get("SVM", 0.0))

            # 1. 结论
            st.metric(f"{best_model_name} {t['prob_label']}", f"{best_prob:.1%}")
            if best_prob > 0.5:
                st.warning(t["high_risk"])
            else:
                st.info(t["low_risk"])

            # 2. 图表
            fig_best = go.Figure(go.Indicator(
                mode="gauge+number",
                value=best_prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"{best_model_name} {t['prob_label']}"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "#ff4b4b" if best_prob > 0.5 else "#28a745"},
                    'steps': [{'range': [0, 0.5], 'color': "lightgray"}, {'range': [0.5, 1], 'color': "gray"}],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.5}
                }
            ))
            fig_best.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_best, use_column_width=True)

        # ---------- 综合模型 ----------
        with tab2:
            st.metric(t["avg_risk"], f"{weighted_avg_risk:.1%}")
            if weighted_avg_risk > 0.5:
                st.warning(t["high_risk"])
            else:
                st.info(t["low_risk"])

            # 综合模型对比柱状图
            res_df = pd.DataFrame({"Algorithm": list(results.keys()), "Prob": list(results.values())}).sort_values(
                "Prob")
            fig_all = go.Figure(go.Bar(
                x=res_df["Prob"],
                y=res_df["Algorithm"],
                orientation='h',
                marker=dict(color=res_df["Prob"], colorscale='RdYlBu_r')
            ))
            fig_all.update_layout(xaxis_title=t["x_axis"], xaxis=dict(range=[0, 1]), height=400,
                                  margin=dict(l=0, r=0, t=30, b=0))
            fig_all.add_vline(x=0.5, line_dash="dash", line_color="red")
            st.plotly_chart(fig_all, use_column_width=True)

        with st.expander(t["model_info"]):
            selection_auc = model_cfg.get("selection_cv_auc", {}).get(best_model_name)
            if selection_auc is not None:
                st.caption(
                    f"{best_model_name}: training five-fold CV AUC = {selection_auc:.3f}"
                )
            st.caption(t["pipeline_info"])
            if missing_count:
                st.caption(
                    f"{missing_count} input feature(s) were absent after extraction "
                    "and were completed by the fitted training-fold imputation means."
                )
            features = model_cfg.get("final_features", [])
            if features:
                st.markdown(f"**{t['final_features']}**")
                st.write(", ".join(features))

else:
    st.info(t["info"])
