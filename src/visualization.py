"""
Visualisations pour le projet de prédiction de résistance TB
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from pathlib import Path

# Configuration des styles
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def visualize_data_distribution(df, output_dir='data/results'):
    """
    Visualiser la distribution des données
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribution par pays
    if 'country' in df.columns:
        country_counts = df['country'].value_counts().head(10)
        country_counts.plot(kind='bar', ax=axes[0,0], color='steelblue')
        axes[0,0].set_title('Top 10 Countries', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Country', fontsize=12)
        axes[0,0].set_ylabel('Number of Isolates', fontsize=12)
        axes[0,0].tick_params(axis='x', rotation=45)
    else:
        axes[0,0].text(0.5, 0.5, 'Country data not available', 
                      ha='center', va='center', transform=axes[0,0].transAxes)
        axes[0,0].set_title('Country Distribution')
    
    # 2. Distribution des lignées
    if 'lineage' in df.columns:
        lineage_counts = df['lineage'].value_counts()
        lineage_counts.plot(kind='bar', ax=axes[0,1], color='coral')
        axes[0,1].set_title('Lineage Distribution', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Lineage', fontsize=12)
        axes[0,1].set_ylabel('Count', fontsize=12)
        axes[0,1].tick_params(axis='x', rotation=45)
    else:
        axes[0,1].text(0.5, 0.5, 'Lineage data not available', 
                      ha='center', va='center', transform=axes[0,1].transAxes)
        axes[0,1].set_title('Lineage Distribution')
    
    # 3. Distribution des profils de résistance
    if 'resistance_profile' in df.columns:
        resistance_counts = df['resistance_profile'].value_counts()
        resistance_counts.plot(kind='pie', ax=axes[1,0], autopct='%1.1f%%', startangle=90)
        axes[1,0].set_title('Resistance Profile Distribution', fontsize=14, fontweight='bold')
        axes[1,0].set_ylabel('')
    else:
        axes[1,0].text(0.5, 0.5, 'Resistance profile data not available', 
                      ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Resistance Profile Distribution')
    
    # 4. Mutations par gène (si disponible)
    resistance_genes = ['rpoB', 'katG', 'inhA', 'embB', 'pncA', 'gyrA']
    gene_mutations = {}
    for gene in resistance_genes:
        gene_cols = [col for col in df.columns if gene.lower() in col.lower()]
        if gene_cols:
            gene_mutations[gene] = df[gene_cols].sum().sum()
    
    if gene_mutations:
        pd.Series(gene_mutations).plot(kind='bar', ax=axes[1,1], color='green')
        axes[1,1].set_title('Total Mutations per Gene', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Gene', fontsize=12)
        axes[1,1].set_ylabel('Number of Mutations', fontsize=12)
        axes[1,1].tick_params(axis='x', rotation=45)
    else:
        axes[1,1].text(0.5, 0.5, 'Mutation data not available', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Mutations per Gene')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/data_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/data_distribution.png")
    plt.close()

def visualize_feature_importance(feature_importance_df, top_n=20, output_dir='data/results'):
    """
    Visualiser l'importance des features
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    top_features = feature_importance_df.head(top_n)
    
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    plt.title(f'Top {top_n} Most Important Features', fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/feature_importance.png")
    plt.close()

def visualize_model_comparison(results_dict, output_dir='data/results'):
    """
    Comparer les performances des modèles
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    models = list(results_dict.keys())
    
    # Extraire les scores
    cv_scores = []
    test_scores = []
    test_accuracies = []
    
    for model in models:
        result = results_dict[model]
        if result.get('cv_mean_auc') is not None:
            cv_scores.append(result['cv_mean_auc'])
        elif result.get('cv_mean_accuracy') is not None:
            cv_scores.append(result['cv_mean_accuracy'])
        else:
            cv_scores.append(0)
        
        if result.get('test_auc') is not None:
            test_scores.append(result['test_auc'])
        else:
            test_scores.append(result.get('test_accuracy', 0))
        
        test_accuracies.append(result.get('test_accuracy', 0))
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, cv_scores, width, label='CV Score', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, test_scores, width, label='Test Score', alpha=0.8, color='coral')
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/model_comparison.png")
    plt.close()

def plot_roc_curves(models_dict, X_test, y_test, output_dir='data/results'):
    """
    Tracer les courbes ROC pour tous les modèles
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    for model_name, model in models_dict.items():
        try:
            y_pred_proba = model.predict_proba(X_test)
            
            if len(np.unique(y_test)) == 2:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                # Multi-class ROC
                from sklearn.preprocessing import label_binarize
                from sklearn.metrics import auc as sklearn_auc
                from itertools import cycle
                
                y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
                n_classes = y_test_bin.shape[1]
                
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                    roc_auc[i] = sklearn_auc(fpr[i], tpr[i])
                
                # Moyenne macro
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(n_classes):
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                mean_tpr /= n_classes
                
                fpr = all_fpr
                tpr = mean_tpr
                auc_score = np.mean(list(roc_auc.values()))
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
        except Exception as e:
            print(f"Error plotting ROC for {model_name}: {e}")
            continue
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/roc_curves.png")
    plt.close()

