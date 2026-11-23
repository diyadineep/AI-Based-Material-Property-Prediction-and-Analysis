"""
AI-Based Material Property Prediction and Analysis ("Compound Explorer")
Streamlit Web Application
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.feature_extractor import CompositionFeatureExtractor
from src.similarity import MaterialSimilarityEngine
from src.explainability import ExplainabilityEngine

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Compound Explorer | AI Material Property Predictor",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Modern CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.4);
    }
    
    .main-header h1 {
        color: #ffffff;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #c7d2fe;
        font-size: 1.1rem;
    }
    
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.5rem 0;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .badge-metal {
        background-color: #0284c7;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1rem;
        display: inline-block;
    }
    
    .badge-nonmetal {
        background-color: #059669;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1rem;
        display: inline-block;
    }
    
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')


@st.cache_resource
def load_all_artifacts():
    """Load cached model artifacts, scalers, and metrics."""
    try:
        reg_scaler = joblib.load(os.path.join(MODEL_DIR, 'regression_scaler.pkl'))
        best_reg = joblib.load(os.path.join(MODEL_DIR, 'best_regression_model.pkl'))
        all_reg = joblib.load(os.path.join(MODEL_DIR, 'all_regression_models.pkl'))

        clf_scaler = joblib.load(os.path.join(MODEL_DIR, 'classification_scaler.pkl'))
        best_clf = joblib.load(os.path.join(MODEL_DIR, 'best_classification_model.pkl'))
        all_clf = joblib.load(os.path.join(MODEL_DIR, 'all_classification_models.pkl'))

        feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
        dataset_mean_feats = joblib.load(os.path.join(MODEL_DIR, 'dataset_mean_features.pkl'))

        with open(os.path.join(MODEL_DIR, 'regression_metrics.json')) as f:
            reg_metrics = json.load(f)

        with open(os.path.join(MODEL_DIR, 'classification_metrics.json')) as f:
            clf_metrics = json.load(f)

        sim_engine = MaterialSimilarityEngine(metric='cosine')
        sim_engine.load_artifacts()

        return {
            'reg_scaler': reg_scaler,
            'best_reg': best_reg,
            'all_reg': all_reg,
            'clf_scaler': clf_scaler,
            'best_clf': best_clf,
            'all_clf': all_clf,
            'feature_names': feature_names,
            'dataset_mean_feats': dataset_mean_feats,
            'reg_metrics': reg_metrics,
            'clf_metrics': clf_metrics,
            'sim_engine': sim_engine
        }
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}. Please run `python run_pipeline.py` first.")
        return None


def main():
    # Header Banner
    st.markdown("""
    <div class="main-header">
        <h1>⚛️ Compound Explorer</h1>
        <p>AI-Based Material Property Prediction, Similarity Retrieval & Explainable ML Framework</p>
    </div>
    """, unsafe_allow_html=True)

    artifacts = load_all_artifacts()
    if not artifacts:
        st.warning("⚠️ Pipeline artifacts not found. Run `python run_pipeline.py` to train models and generate artifacts.")
        return

    extractor = CompositionFeatureExtractor()

    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/color/96/000000/molecule.png", width=64)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Module:",
        [
            "🔮 Predictor & Recommendation",
            "📊 Model Benchmarks & Tuning",
            "🧠 Feature Importance & SHAP",
            "🔬 Matbench Dataset Explorer"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Dataset**: Matbench Curie & Experimental Datasets
    - `matbench_expt_gap` (4,604 samples)
    - `matbench_expt_is_metal` (4,921 samples)
    """)

    # -------------------------------------------------------------------------
    # TAB 1: PREDICTOR & RECOMMENDATION
    # -------------------------------------------------------------------------
    if page == "🔮 Predictor & Recommendation":
        st.subheader("Predict Material Properties & Discover Similar Compounds")

        col_input, col_preset = st.columns([2, 1])

        with col_input:
            input_formula = st.text_input(
                "Enter Chemical Formula:",
                value="TiO2",
                placeholder="e.g. TiO2, GaAs, BaTiO3, Fe2O3, CsPbI3",
                help="Type a valid chemical formula string."
            )

        with col_preset:
            st.write("Quick Presets:")
            preset_cols = st.columns(4)
            preset_click = None
            if preset_cols[0].button("TiO₂"): preset_click = "TiO2"
            if preset_cols[1].button("GaAs"): preset_click = "GaAs"
            if preset_cols[2].button("BaTiO₃"): preset_click = "BaTiO3"
            if preset_cols[3].button("CsPbI₃"): preset_click = "CsPbI3"

            if preset_click:
                input_formula = preset_click

        if input_formula:
            try:
                # Extract features for query
                query_feat_dict = extractor.featurize_formula(input_formula)
                query_df = pd.DataFrame([query_feat_dict])[artifacts['feature_names']]

                # Scaled features
                reg_scaled = artifacts['reg_scaler'].transform(query_df.values)
                clf_scaled = artifacts['clf_scaler'].transform(query_df.values)

                # Predictions
                pred_gap = float(artifacts['best_reg'].predict(reg_scaled)[0])
                pred_gap = max(0.0, pred_gap)  # band gap >= 0

                pred_metal_cls = int(artifacts['best_clf'].predict(clf_scaled)[0])
                pred_metal_proba = float(artifacts['best_clf'].predict_proba(clf_scaled)[0][1]) if hasattr(artifacts['best_clf'], 'predict_proba') else 0.5

                st.markdown("---")
                # Results Banner Cards
                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div>Experimental Band Gap (Predicted)</div>
                        <div class="metric-val">{pred_gap:.2f} eV</div>
                        <small style="color: #94a3b8;">Model: {artifacts['reg_metrics']['best_model_name']}</small>
                    </div>
                    """, unsafe_allow_html=True)

                with c2:
                    is_metal_str = "METALLIC" if pred_metal_cls == 1 else "NON-METALLIC"
                    badge_class = "badge-metal" if pred_metal_cls == 1 else "badge-nonmetal"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div>Metallicity Classification</div>
                        <div style="margin: 0.8rem 0;"><span class="{badge_class}">{is_metal_str}</span></div>
                        <small style="color: #94a3b8;">Confidence: {max(pred_metal_proba, 1-pred_metal_proba)*100:.1f}%</small>
                    </div>
                    """, unsafe_allow_html=True)

                with c3:
                    n_els = query_feat_dict.get('n_elements', 0)
                    total_atoms = query_feat_dict.get('total_atoms', 0)
                    mean_x = query_feat_dict.get('X_mean', 0)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div>Composition Descriptors</div>
                        <div style="font-size: 1.1rem; margin: 0.6rem 0; font-weight: 600;">
                            Elements: {n_els} | Atoms: {int(total_atoms)}
                        </div>
                        <small style="color: #94a3b8;">Mean Electronegativity: {mean_x:.2f}</small>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")

                # Similar Materials & Explainability Split
                res_col1, res_col2 = st.columns([1, 1])

                with res_col1:
                    st.subheader("🔍 Top-5 Similar Materials")
                    st.caption("Retrieval based on composition-derived feature distance from Matbench")
                    df_sim = artifacts['sim_engine'].find_similar_materials(query_feat_dict, top_k=5)
                    st.dataframe(df_sim, use_container_width=True, hide_index=True)

                with res_col2:
                    st.subheader("🧠 Local Prediction Explainability")
                    st.caption(f"Feature contribution relative to dataset average")

                    explainer_engine = ExplainabilityEngine(artifacts['best_reg'], artifacts['feature_names'])
                    df_local, rationale = explainer_engine.generate_local_explanation(
                        query_feat_dict,
                        artifacts['dataset_mean_feats'],
                        top_n=6
                    )

                    fig_local = px.bar(
                        df_local,
                        x='Difference',
                        y='Display Name',
                        orientation='h',
                        color='Difference',
                        color_continuous_scale='Spectral',
                        title="Key Descriptor Deviations"
                    )
                    fig_local.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig_local, use_container_width=True)

                    with st.expander("📖 View Chemical Rationale Summary", expanded=True):
                        st.markdown(rationale)

            except Exception as e:
                st.error(f"Error parsing formula '{input_formula}': {e}")

    # -------------------------------------------------------------------------
    # TAB 2: MODEL BENCHMARKS & TUNING
    # -------------------------------------------------------------------------
    elif page == "📊 Model Benchmarks & Tuning":
        st.subheader("Model Comparison & GridSearchCV Hyperparameter Optimization")

        t1, t2 = st.tabs(["📉 Band Gap Regression Models", "⚡ Metallicity Classification Models"])

        with t1:
            st.markdown("### Band Gap Regression Benchmark (5-Fold CV)")
            reg_metrics = artifacts['reg_metrics']

            df_base_reg = pd.DataFrame(reg_metrics['base_results']).T
            df_tuned_reg = pd.DataFrame(reg_metrics['tuned_results']).T

            c_r1, c_r2 = st.columns(2)

            with c_r1:
                st.write("**Default Models Performance:**")
                st.dataframe(df_base_reg.style.format("{:.4f}").highlight_min(subset=['MAE', 'RMSE'], color='#1e3a8a').highlight_max(subset=['R2'], color='#1e3a8a'), use_container_width=True)

            with c_r2:
                st.write("**GridSearchCV Optimized Performance:**")
                st.dataframe(df_tuned_reg.style.format("{:.4f}").highlight_min(subset=['MAE', 'RMSE'], color='#065f46').highlight_max(subset=['R2'], color='#065f46'), use_container_width=True)

            # R2 Comparison Chart
            df_r2_comp = pd.DataFrame({
                'Model': df_base_reg.index,
                'Default R²': df_base_reg['R2'],
                'GridSearchCV R²': df_tuned_reg['R2']
            }).melt(id_vars=['Model'], var_name='Configuration', value_name='R2 Score')

            fig_r2 = px.bar(
                df_r2_comp,
                x='Model',
                y='R2 Score',
                color='Configuration',
                barmode='group',
                title="Hyperparameter Optimization Impact on R² Score",
                color_discrete_sequence=['#94a3b8', '#6366f1']
            )
            st.plotly_chart(fig_r2, use_container_width=True)

            st.write(f"🏆 **Best Regression Model Selected**: `{reg_metrics['best_model_name']}`")
            st.json(reg_metrics['best_params'][reg_metrics['best_model_name']])

        with t2:
            st.markdown("### Metallicity Classification Benchmark (5-Fold Stratified CV)")
            clf_metrics = artifacts['clf_metrics']

            df_base_clf = pd.DataFrame(clf_metrics['base_results']).T
            df_tuned_clf = pd.DataFrame(clf_metrics['tuned_results']).T

            c_c1, c_c2 = st.columns(2)

            with c_c1:
                st.write("**Default Classifiers Performance:**")
                st.dataframe(df_base_clf.style.format("{:.4f}").highlight_max(subset=['Accuracy', 'F1'], color='#1e3a8a'), use_container_width=True)

            with c_c2:
                st.write("**GridSearchCV Optimized Performance:**")
                st.dataframe(df_tuned_clf.style.format("{:.4f}").highlight_max(subset=['Accuracy', 'F1', 'ROC_AUC'], color='#065f46'), use_container_width=True)

            # Confusion Matrix & ROC Curve
            cm_col, roc_col = st.columns(2)

            with cm_col:
                best_clf_name = clf_metrics['best_model_name']
                cm = np.array(clf_metrics['confusion_matrices'][best_clf_name])
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    color_continuous_scale='Blues',
                    labels=dict(x="Predicted Label", y="True Label"),
                    x=['Non-Metal', 'Metal'],
                    y=['Non-Metal', 'Metal'],
                    title=f"Confusion Matrix ({best_clf_name})"
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            with roc_col:
                fig_roc = go.Figure()
                for name, curve_data in clf_metrics['roc_curves'].items():
                    fig_roc.add_trace(go.Scatter(
                        x=curve_data['fpr'],
                        y=curve_data['tpr'],
                        mode='lines',
                        name=f"{name} (AUC = {curve_data['auc']:.3f})"
                    ))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Random Chance'))
                fig_roc.update_layout(title="ROC-AUC Curves", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(fig_roc, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: FEATURE IMPORTANCE & SHAP
    # -------------------------------------------------------------------------
    elif page == "🧠 Feature Importance & SHAP":
        st.subheader("Global Feature Importance & Model Interpretability")

        explainer_reg = ExplainabilityEngine(artifacts['best_reg'], artifacts['feature_names'])
        df_imp_reg = explainer_reg.get_global_feature_importance(top_n=15)

        explainer_clf = ExplainabilityEngine(artifacts['best_clf'], artifacts['feature_names'])
        df_imp_clf = explainer_clf.get_global_feature_importance(top_n=15)

        col_f1, col_f2 = st.columns(2)

        with col_f1:
            st.markdown("### Top Descriptors: Band Gap Regressor")
            fig_imp_r = px.bar(
                df_imp_reg.sort_values('Importance', ascending=True),
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='Viridis',
                title=f"Gini Importance ({artifacts['reg_metrics']['best_model_name']})"
            )
            fig_imp_r.update_layout(height=500)
            st.plotly_chart(fig_imp_r, use_container_width=True)

        with col_f2:
            st.markdown("### Top Descriptors: Metallicity Classifier")
            fig_imp_c = px.bar(
                df_imp_clf.sort_values('Importance', ascending=True),
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='Plasma',
                title=f"Gini Importance ({artifacts['clf_metrics']['best_model_name']})"
            )
            fig_imp_c.update_layout(height=500)
            st.plotly_chart(fig_imp_c, use_container_width=True)

        st.markdown("---")
        st.subheader("💡 Materials Science Domain Insights")
        st.markdown("""
        - **Electronegativity Statistics (`X_mean`, `X_range`, `X_std`)**: Consistently rank as top predictors. High electronegativity contrast between cations and anions strongly correlates with wider band gaps.
        - **Atomic Radius & Mendeleev Number**: Govern orbital overlap integrals and spatial band structure width. Larger atomic radii lead to lower band gap values.
        - **d-Block & f-Block Fractions**: Transition metal and rare-earth element fractions strongly dictate metallic vs non-metallic ground states due to partially filled d/f shells.
        """)

    # -------------------------------------------------------------------------
    # TAB 4: MATBENCH DATASET EXPLORER
    # -------------------------------------------------------------------------
    elif page == "🔬 Matbench Dataset Explorer":
        st.subheader("Explore Matbench Experimental Datasets")

        df_dataset = artifacts['sim_engine'].dataset_df

        st.write(f"Total Cached Dataset Samples: **{len(df_dataset):,}**")

        st.dataframe(df_dataset.head(100), use_container_width=True, hide_index=True)

        col_d1, col_d2 = st.columns(2)

        with col_d1:
            if 'gap_expt' in df_dataset:
                fig_hist = px.histogram(
                    df_dataset,
                    x='gap_expt',
                    nbins=40,
                    title="Experimental Band Gap Distribution (eV)",
                    color_discrete_sequence=['#6366f1']
                )
                fig_hist.update_layout(xaxis_title="Band Gap (eV)", yaxis_title="Material Count")
                st.plotly_chart(fig_hist, use_container_width=True)

        with col_d2:
            if 'is_metal' in df_dataset:
                df_metal_counts = df_dataset['is_metal'].value_counts().reset_index()
                df_metal_counts.columns = ['Status', 'Count']
                df_metal_counts['Status'] = df_metal_counts['Status'].map({True: 'Metallic', False: 'Non-Metallic'})

                fig_pie = px.pie(
                    df_metal_counts,
                    names='Status',
                    values='Count',
                    title="Metallicity Class Proportion",
                    color_discrete_sequence=['#0284c7', '#059669']
                )
                st.plotly_chart(fig_pie, use_container_width=True)


if __name__ == '__main__':
    main()
