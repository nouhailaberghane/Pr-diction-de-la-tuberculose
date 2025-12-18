"""
Modèles de Machine Learning pour la prédiction de résistance
"""
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

def prepare_ml_data(df, target_col, selected_features):
    """
    Préparer les données pour l'apprentissage machine
    """
    # S'assurer que les features sélectionnées existent
    available_features = [f for f in selected_features if f in df.columns]
    
    if len(available_features) == 0:
        print("Warning: No selected features found in dataframe")
        # Utiliser toutes les colonnes numériques sauf la target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        available_features = numeric_cols[:100]  # Limiter à 100 features
    
    X = df[available_features]
    y = df[target_col]
    
    # Encoder la target si nécessaire
    le = None
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Standardiser les features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertir en DataFrame pour garder les noms de colonnes
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=available_features, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=available_features, index=X_test.index)
    
    return (X_train_scaled, X_test_scaled, y_train, y_test, scaler, le, available_features)

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Modèle 1: Logistic Regression
    """
    print("\nTraining Logistic Regression...")
    
    # Hyperparameter tuning simplifié pour éviter les temps d'exécution trop longs
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    lr = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
    
    try:
        grid_search = GridSearchCV(lr, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)
        best_lr = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    except Exception as e:
        print(f"Grid search failed: {e}. Using default parameters.")
        best_lr = LogisticRegression(random_state=42, max_iter=1000, C=1.0, penalty='l2', solver='liblinear')
        best_lr.fit(X_train, y_train)
    
    y_pred = best_lr.predict(X_test)
    y_pred_proba = best_lr.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) == 2 else best_lr.predict_proba(X_test)
    
    return best_lr, y_pred, y_pred_proba

def train_knn(X_train, y_train, X_test, y_test):
    """
    Modèle 2: K-Nearest Neighbors
    """
    print("\nTraining KNN...")
    
    param_grid = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    knn = KNeighborsClassifier(n_jobs=-1)
    
    try:
        grid_search = GridSearchCV(knn, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)
        best_knn = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    except Exception as e:
        print(f"Grid search failed: {e}. Using default parameters.")
        best_knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        best_knn.fit(X_train, y_train)
    
    y_pred = best_knn.predict(X_test)
    y_pred_proba = best_knn.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) == 2 else best_knn.predict_proba(X_test)
    
    return best_knn, y_pred, y_pred_proba

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Modèle 3: Random Forest
    """
    print("\nTraining Random Forest...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    try:
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)
        best_rf = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    except Exception as e:
        print(f"Grid search failed: {e}. Using default parameters.")
        best_rf = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
        best_rf.fit(X_train, y_train)
    
    y_pred = best_rf.predict(X_test)
    y_pred_proba = best_rf.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) == 2 else best_rf.predict_proba(X_test)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return best_rf, y_pred, y_pred_proba, feature_importance

def evaluate_models(models_dict, X_train, y_train, X_test, y_test):
    """
    Évaluer tous les modèles avec validation croisée
    """
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, model in models_dict.items():
        print(f"\nEvaluating {model_name}...")
        
        # Cross-validation
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        except:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        
        # Test set evaluation
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Métriques
        try:
            if len(np.unique(y_test)) == 2:
                test_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                test_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            test_auc = None
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[model_name] = {
            'cv_mean_auc': cv_scores.mean() if test_auc is not None else None,
            'cv_std_auc': cv_scores.std() if test_auc is not None else None,
            'cv_mean_accuracy': cv_scores.mean() if test_auc is None else None,
            'cv_std_accuracy': cv_scores.std() if test_auc is None else None,
            'test_auc': test_auc,
            'test_accuracy': accuracy,
            'test_f1': f1,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        if test_auc:
            print(f"Test AUC: {test_auc:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test F1: {f1:.4f}")
    
    return results

