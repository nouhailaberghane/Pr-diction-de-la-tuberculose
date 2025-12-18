"""
Feature selection pour le dataset Afro-TB
"""
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    print("Warning: mlxtend not available. Pattern recognition will be skipped.")

def remove_synonymous_features(df, target_col):
    """
    Étape 4.1: Supprimer les features synonymes
    (mutations qui donnent le même résultat)
    """
    # Identifier les features avec corrélation parfaite
    # Utiliser seulement les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    if len(numeric_cols) == 0:
        print("No numeric columns found for correlation analysis")
        return df, []
    
    # Calculer la matrice de corrélation
    correlation_matrix = df[numeric_cols].corr().abs()
    
    # Trouver les paires de features avec corrélation > 0.95
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if corr_val > 0.95 and not np.isnan(corr_val):
                high_corr_pairs.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j]
                ))
    
    # Supprimer une feature de chaque paire (garder celle avec meilleure corrélation avec target)
    features_to_remove = set()
    
    for feat1, feat2 in high_corr_pairs:
        if target_col in df.columns:
            try:
                corr1 = abs(df[feat1].corr(df[target_col]))
                corr2 = abs(df[feat2].corr(df[target_col]))
                
                if pd.isna(corr1) or pd.isna(corr2):
                    # Si corrélation non calculable, garder la première
                    features_to_remove.add(feat2)
                elif corr1 < corr2:
                    features_to_remove.add(feat1)
                else:
                    features_to_remove.add(feat2)
            except:
                # En cas d'erreur, garder la première
                features_to_remove.add(feat2)
        else:
            # Si pas de target, garder la première
            features_to_remove.add(feat2)
    
    df_filtered = df.drop(columns=list(features_to_remove))
    
    print(f"Removed {len(features_to_remove)} synonymous features")
    
    return df_filtered, list(features_to_remove)

def select_highly_correlated_features(df, target_col, threshold=0.1):
    """
    Étape 4.2: Sélectionner les features hautement corrélées avec la résistance
    """
    if target_col not in df.columns:
        print(f"Warning: Target column '{target_col}' not found")
        # Retourner toutes les colonnes numériques sauf la target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return numeric_cols
    
    # Utiliser seulement les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    if len(numeric_cols) == 0:
        print("No numeric columns found")
        return []
    
    # Calculer la corrélation avec la target
    correlations = {}
    for col in numeric_cols:
        try:
            corr = abs(df[col].corr(df[target_col]))
            if not pd.isna(corr):
                correlations[col] = corr
        except:
            continue
    
    if len(correlations) == 0:
        print("Could not calculate correlations")
        return numeric_cols  # Retourner toutes les colonnes numériques
    
    # Sélectionner les features avec corrélation > threshold
    selected_features = [col for col, corr in correlations.items() if corr > threshold]
    
    print(f"Selected {len(selected_features)} features with correlation > {threshold}")
    
    return selected_features

def pattern_recognition_apriori(df, min_support=0.01):
    """
    Étape 4.3: Pattern recognition avec algorithme Apriori
    Pour identifier les combinaisons de mutations fréquentes
    """
    if not MLXTEND_AVAILABLE:
        print("mlxtend not available. Skipping pattern recognition.")
        return pd.DataFrame(), pd.DataFrame()
    
    # Convertir en format binaire pour Apriori
    binary_df = df.copy()
    
    # S'assurer que toutes les valeurs sont binaires (0 ou 1)
    for col in binary_df.columns:
        if binary_df[col].dtype in ['int64', 'float64']:
            binary_df[col] = (binary_df[col] > 0).astype(int)
    
    try:
        # Appliquer Apriori
        frequent_itemsets = apriori(
            binary_df, 
            min_support=min_support, 
            use_colnames=True,
            max_len=3  # Limiter la longueur pour éviter les combinaisons trop complexes
        )
        
        # Générer les règles d'association
        if len(frequent_itemsets) > 0:
            rules = association_rules(
                frequent_itemsets, 
                metric="confidence", 
                min_threshold=0.5
            )
            
            # Filtrer les règles liées à la résistance
            if 'resistant' in str(rules.get('consequents', pd.Series())):
                resistance_rules = rules[
                    rules['consequents'].astype(str).str.contains('resistant', case=False, na=False)
                ]
                return frequent_itemsets, resistance_rules
            
            return frequent_itemsets, rules
        
        return frequent_itemsets, pd.DataFrame()
        
    except Exception as e:
        print(f"Error in pattern recognition: {e}")
        return pd.DataFrame(), pd.DataFrame()

def select_features_with_chi2(X, y, k=100):
    """
    Sélectionner les meilleures features avec chi-square
    """
    # Convertir en binaire si nécessaire
    X_binary = (X > 0).astype(int)
    
    try:
        selector = SelectKBest(score_func=chi2, k=min(k, X_binary.shape[1]))
        selector.fit(X_binary, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        scores = selector.scores_
        
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': scores
        }).sort_values('score', ascending=False)
        
        return selected_features, feature_scores
    except Exception as e:
        print(f"Error in chi2 selection: {e}")
        return X.columns.tolist(), pd.DataFrame()

def select_features_with_mutual_info(X, y, k=100):
    """
    Sélectionner les meilleures features avec mutual information
    """
    try:
        selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        selector.fit(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        scores = selector.scores_
        
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': scores
        }).sort_values('score', ascending=False)
        
        return selected_features, feature_scores
    except Exception as e:
        print(f"Error in mutual info selection: {e}")
        return X.columns.tolist(), pd.DataFrame()

