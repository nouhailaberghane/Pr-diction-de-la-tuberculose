"""
Script principal pour le pipeline complet de prédiction de résistance TB
"""
import sys
import os
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.append(str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
from data_loader import AfroTBDataLoader
from data_cleaner import DataCleaner
from feature_extractor import MutationFeatureExtractor
from target_creator import ResistanceProfileCreator
from feature_selection import (
    remove_synonymous_features,
    select_highly_correlated_features,
    pattern_recognition_apriori,
    select_features_with_chi2,
    select_features_with_mutual_info
)
from models import (
    prepare_ml_data,
    train_logistic_regression,
    train_knn,
    train_random_forest,
    evaluate_models
)
from visualization import (
    visualize_data_distribution,
    visualize_feature_importance,
    visualize_model_comparison,
    plot_roc_curves
)
from fitness_analysis import FitnessMutationAnalyzer

def main():
    """
    Pipeline complet du projet
    """
    print("="*60)
    print("TUBERCULOSIS DRUG RESISTANCE PREDICTION")
    print("="*60)
    
    # Créer les répertoires nécessaires
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path('data/results').mkdir(parents=True, exist_ok=True)
    
    # 1. Charger les données
    print("\n[1/7] Loading dataset...")
    try:
        loader = AfroTBDataLoader('data/raw')
        df = loader.load_dataset()
        print(f"Loaded {len(df)} isolates")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease download the dataset first:")
        print("1. Run: python src/data_download.py")
        print("2. Or download manually from:")
        print("   https://springernature.figshare.com/articles/dataset/Afro-TB_dataset/21803712")
        return None
    
    # Explorer le dataset
    loader.explore_dataset()
    
    # 2. Nettoyer les données
    print("\n[2/7] Cleaning dataset...")
    cleaner = DataCleaner(df)
    df_clean = cleaner.clean_dataset()
    
    # Sauvegarder les données nettoyées
    df_clean.to_csv('data/processed/cleaned_dataset.csv', index=False)
    print("Saved cleaned dataset to: data/processed/cleaned_dataset.csv")
    
    # 3. Extraire les features de mutations
    print("\n[3/7] Extracting mutation features...")
    extractor = MutationFeatureExtractor(df_clean)
    mutation_features = extractor.extract_mutation_features()
    
    # Features au niveau gène
    gene_features = extractor.create_gene_level_features()
    
    # Features au niveau médicament
    drug_features = extractor.create_drug_resistance_features()
    
    # Sauvegarder
    if len(mutation_features.columns) > 0:
        mutation_features.to_csv('data/processed/mutation_features.csv', index=False)
    if len(gene_features.columns) > 0:
        gene_features.to_csv('data/processed/gene_features.csv', index=False)
    if len(drug_features.columns) > 0:
        drug_features.to_csv('data/processed/drug_features.csv', index=False)
    
    print(f"Extracted {len(mutation_features.columns)} mutation features")
    print(f"Extracted {len(gene_features.columns)} gene-level features")
    print(f"Extracted {len(drug_features.columns)} drug-level features")
    
    # 4. Créer les profils de résistance
    print("\n[4/7] Creating resistance profiles...")
    profile_creator = ResistanceProfileCreator(df_clean, mutation_features)
    df_final = profile_creator.create_resistance_profiles()
    
    # 5. Fusionner toutes les features
    print("\n[5/7] Merging all features...")
    
    # Préparer les DataFrames à fusionner
    merge_list = []
    
    # Métadonnées
    metadata_cols = ['sample_id', 'country', 'lineage']
    metadata_cols = [col for col in metadata_cols if col in df_final.columns]
    if metadata_cols:
        merge_list.append(df_final[metadata_cols])
    
    # Mutations
    if len(mutation_features.columns) > 0:
        merge_list.append(mutation_features)
    
    # Features gènes
    if len(gene_features.columns) > 0:
        merge_list.append(gene_features)
    
    # Features médicaments
    if len(drug_features.columns) > 0:
        merge_list.append(drug_features)
    
    # Targets
    target_cols = ['resistance_profile', 'resistant_rifampicin', 
                   'resistant_isoniazid', 'resistant_ethambutol',
                   'resistant_pyrazinamide', 'resistant_fluoroquinolones']
    target_cols = [col for col in target_cols if col in df_final.columns]
    if target_cols:
        merge_list.append(df_final[target_cols])
    
    # Fusionner
    if len(merge_list) > 0:
        df_ml_ready = pd.concat(merge_list, axis=1)
    else:
        df_ml_ready = df_final.copy()
    
    # Sauvegarder le dataset final
    df_ml_ready.to_csv('data/processed/ml_ready_dataset.csv', index=False)
    print("Saved ML-ready dataset to: data/processed/ml_ready_dataset.csv")
    
    # Résumé final
    print("\n" + "="*60)
    print("DATA PREPARATION SUMMARY")
    print("="*60)
    print(f"Total isolates: {len(df_ml_ready)}")
    print(f"Total features: {len(df_ml_ready.columns)}")
    print(f"Mutation features: {len(mutation_features.columns)}")
    print(f"Gene features: {len(gene_features.columns)}")
    print(f"Drug features: {len(drug_features.columns)}")
    if 'resistance_profile' in df_ml_ready.columns:
        print(f"\nResistance profiles:")
        print(df_ml_ready['resistance_profile'].value_counts())
    
    # 6. Feature Selection
    print("\n[6/7] Feature selection...")
    
    # Déterminer la colonne target
    if 'resistant_rifampicin' in df_ml_ready.columns:
        target_col = 'resistant_rifampicin'  # Classification binaire
    elif 'resistance_profile' in df_ml_ready.columns:
        target_col = 'resistance_profile'  # Classification multi-classe
    else:
        print("Warning: No target column found. Using first resistance column.")
        target_col = [col for col in df_ml_ready.columns if 'resistant' in col.lower()][0]
    
    print(f"Target column: {target_col}")
    
    # Préparer les features (exclure les colonnes non-numériques et la target)
    feature_cols = [col for col in df_ml_ready.columns 
                   if col != target_col and 
                   col not in ['sample_id', 'country', 'lineage', 'sra_accession']]
    
    # Utiliser seulement les colonnes numériques
    numeric_cols = df_ml_ready[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        print("Warning: No numeric features found. Using all columns.")
        numeric_cols = feature_cols
    
    df_features = df_ml_ready[numeric_cols + [target_col]].copy()
    
    # Supprimer les features synonymes
    df_filtered, removed = remove_synonymous_features(df_features, target_col)
    
    # Sélectionner les features corrélées
    selected_features = select_highly_correlated_features(df_filtered, target_col, threshold=0.05)
    
    # Si pas assez de features, utiliser toutes les colonnes numériques
    if len(selected_features) < 10:
        print("Warning: Too few features selected. Using all numeric features.")
        selected_features = [col for col in numeric_cols if col in df_filtered.columns]
        selected_features = selected_features[:200]  # Limiter à 200 features
    
    print(f"Final selected features: {len(selected_features)}")
    
    # Pattern recognition (optionnel)
    try:
        frequent_itemsets, rules = pattern_recognition_apriori(
            df_filtered[selected_features[:50] + [target_col]],  # Limiter pour éviter les problèmes de mémoire
            min_support=0.01
        )
        if len(frequent_itemsets) > 0:
            print(f"Found {len(frequent_itemsets)} frequent itemsets")
        if len(rules) > 0:
            print(f"Found {len(rules)} association rules")
    except Exception as e:
        print(f"Pattern recognition skipped: {e}")
    
    # 7. Préparation ML
    print("\n[7/7] Preparing ML data...")
    X_train, X_test, y_train, y_test, scaler, le, final_features = prepare_ml_data(
        df_filtered, target_col, selected_features
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # 8. Entraînement des modèles
    print("\n" + "="*60)
    print("TRAINING MACHINE LEARNING MODELS")
    print("="*60)
    
    models_dict = {}
    feature_importance = None
    
    # Logistic Regression
    try:
        lr_model, lr_pred, lr_proba = train_logistic_regression(
            X_train, y_train, X_test, y_test
        )
        models_dict['Logistic Regression'] = lr_model
    except Exception as e:
        print(f"Error training Logistic Regression: {e}")
    
    # KNN
    try:
        knn_model, knn_pred, knn_proba = train_knn(
            X_train, y_train, X_test, y_test
        )
        models_dict['KNN'] = knn_model
    except Exception as e:
        print(f"Error training KNN: {e}")
    
    # Random Forest
    try:
        rf_model, rf_pred, rf_proba, feature_importance = train_random_forest(
            X_train, y_train, X_test, y_test
        )
        models_dict['Random Forest'] = rf_model
    except Exception as e:
        print(f"Error training Random Forest: {e}")
    
    if len(models_dict) == 0:
        print("Error: No models were trained successfully.")
        return None
    
    # 9. Évaluation
    print("\n" + "="*60)
    print("EVALUATING MODELS")
    print("="*60)
    results = evaluate_models(models_dict, X_train, y_train, X_test, y_test)
    
    # 10. Visualisations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    try:
        visualize_data_distribution(df_clean)
    except Exception as e:
        print(f"Error in data visualization: {e}")
    
    if feature_importance is not None and len(feature_importance) > 0:
        try:
            visualize_feature_importance(feature_importance)
        except Exception as e:
            print(f"Error in feature importance visualization: {e}")
    
    try:
        visualize_model_comparison(results)
    except Exception as e:
        print(f"Error in model comparison visualization: {e}")
    
    try:
        plot_roc_curves(models_dict, X_test, y_test)
    except Exception as e:
        print(f"Error in ROC curves visualization: {e}")
    
    # 11. Sauvegarder les résultats
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Sauvegarder les résultats des modèles
    results_summary = []
    for model_name, result in results.items():
        results_summary.append({
            'Model': model_name,
            'CV_Mean_Score': result.get('cv_mean_auc') or result.get('cv_mean_accuracy'),
            'CV_Std_Score': result.get('cv_std_auc') or result.get('cv_std_accuracy'),
            'Test_AUC': result.get('test_auc'),
            'Test_Accuracy': result.get('test_accuracy'),
            'Test_F1': result.get('test_f1')
        })
    
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv('data/results/model_results.csv', index=False)
    print("Saved: data/results/model_results.csv")
    
    # Sauvegarder l'importance des features
    if feature_importance is not None and len(feature_importance) > 0:
        feature_importance.to_csv('data/results/feature_importance.csv', index=False)
        print("Saved: data/results/feature_importance.csv")
    
    # Sauvegarder les rapports détaillés
    with open('data/results/classification_reports.txt', 'w') as f:
        for model_name, result in results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"{model_name}\n")
            f.write(f"{'='*60}\n")
            f.write(result['classification_report'])
            f.write("\n\n")
    print("Saved: data/results/classification_reports.txt")
    
    print("\n" + "="*60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nResults saved in: data/results/")
    print("Processed data saved in: data/processed/")
    
    return models_dict, results, feature_importance

if __name__ == "__main__":
    models, results, features = main()

