"""
Full Pipeline Automation Script for AI-Based Material Property Prediction.
Orchestrates:
1. Dataset Loading (matbench_expt_gap & matbench_expt_is_metal)
2. Composition Feature Extraction
3. Regression & Classification Model Training with GridSearchCV 5-Fold CV
4. Evaluation Metrics & Parity/ROC Data Persistence
5. Nearest Neighbors Similarity Indexing
6. Saving all artifacts for Streamlit Web Application
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from src.data_loader import load_bandgap_dataset, load_metallicity_dataset
from src.feature_extractor import CompositionFeatureExtractor
from src.train_models import train_and_optimize_regression, train_and_optimize_classification
from src.similarity import MaterialSimilarityEngine

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    print("=========================================================================")
    print("  AI-BASED MATERIAL PROPERTY PREDICTION & ANALYSIS PIPELINE  ")
    print("=========================================================================")

    # Step 1: Load Datasets
    print("\n[Step 1/5] Loading Matbench Datasets...")
    df_gap = load_bandgap_dataset()
    df_metal = load_metallicity_dataset()
    print(f"Loaded Band Gap Dataset: {df_gap.shape[0]} samples.")
    print(f"Loaded Metallicity Dataset: {df_metal.shape[0]} samples.")

    # Step 2: Extract Composition Descriptors
    print("\n[Step 2/5] Extracting Composition-Based Features...")
    extractor = CompositionFeatureExtractor()

    print("Extracting features for Band Gap dataset...")
    X_gap, valid_idx_gap = extractor.featurize_dataframe(df_gap, formula_col='formula')
    df_gap = df_gap.iloc[valid_idx_gap].reset_index(drop=True)
    y_gap = df_gap['gap_expt'].values

    print("Extracting features for Metallicity dataset...")
    X_metal, valid_idx_metal = extractor.featurize_dataframe(df_metal, formula_col='formula')
    df_metal = df_metal.iloc[valid_idx_metal].reset_index(drop=True)
    y_metal = df_metal['is_metal'].values

    feature_names = list(X_gap.columns)
    print(f"Extracted {len(feature_names)} composition descriptors.")

    # Save feature names and dataset feature means for explainability
    joblib.dump(feature_names, os.path.join(MODEL_DIR, 'feature_names.pkl'))
    dataset_mean_feats = X_gap.mean().to_dict()
    joblib.dump(dataset_mean_feats, os.path.join(MODEL_DIR, 'dataset_mean_features.pkl'))

    # Step 3: Train & Tune Band Gap Regressors
    print("\n[Step 3/5] Training & Optimizing Regression Models...")
    reg_metrics, reg_scaler, best_reg_model = train_and_optimize_regression(X_gap.values, y_gap)

    # Step 4: Train & Tune Metallicity Classifiers
    print("\n[Step 4/5] Training & Optimizing Classification Models...")
    clf_metrics, clf_scaler, best_clf_model = train_and_optimize_classification(X_metal.values, y_metal)

    # Step 5: Fit & Save Similarity Engine
    print("\n[Step 5/5] Building Material Similarity Engine...")
    sim_engine = MaterialSimilarityEngine(metric='cosine')
    sim_engine.fit(df_gap, X_gap.values, feature_names)

    print("\n=========================================================================")
    print("  FULL PIPELINE COMPLETED SUCCESSFULLY!  ")
    print("  All artifacts saved to models/ directory.  ")
    print("=========================================================================")


if __name__ == '__main__':
    main()
