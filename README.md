# Tuberculosis Drug Resistance Prediction

Projet de prédiction de la résistance aux médicaments de *Mycobacterium tuberculosis* à partir du **dataset Afro‑TB** (13 753 isolats africains).

---

## 🎯 Objectifs

- **Analyse de 13 753 isolats de *Mycobacterium tuberculosis* issus de 26 pays africains** pour identifier les mutations responsables de la résistance aux médicaments.
- **Détection de 157 mutations connues dans 12 gènes et identification de nouvelles mutations potentielles liées à la “fitness”** (via les VCF annotés et l’analyse prévue dans `fitness_analysis.py`).
- Exploiter les métadonnées (pays, lignée, profil de résistance) pour prédire la résistance.
- Entraîner et comparer plusieurs modèles de Machine Learning :
  - Logistic Regression
  - K‑Nearest Neighbors (KNN)
  - Random Forest
- Préparer le pipeline pour intégrer les **157 mutations connues dans 12 gènes** (via les fichiers VCF annotés) et l’analyse de “fitness”.

---

## 🧬 Dataset Afro‑TB

- **Source** : Laamarti et al., *Scientific Data* (2023) – « Afro‑TB dataset as a large scale genomic data of *Mycobacterium tuberculosis* in Africa ».
- **Contenu scientifique clé** :
  - 13 753 isolats de *M. tuberculosis* provenant de 26 pays africains.
  - 157 mutations connues dans 12 gènes de résistance, plus des mutations potentielles liées à la fitness (décrites dans l’article et les fichiers VCF annotés).
- **Téléchargement** (déjà fait dans ce projet) :  
  - Afro‑TB sur Figshare : `https://springernature.figshare.com/articles/dataset/Afro-TB_dataset/21803712`
- **Fichiers utilisés** :
  - `data/raw/Afro_TB/0-StartHERE_Afro-TB.xlsx`  
    → fichier principal (13 753 isolats, pays, lignée, profil de drogue, etc.)
  - `data/raw/Afro_TB/AFRO_TB_ANNOTATION_VCF/`  
    → VCF annotés (toutes les mutations détaillées, prêts pour une étape future)

Après nettoyage :

- **Isolats analysés** : 13 691  
- **Pays représentés** : 25  
- **Lignées** : 10 (L1–L6, BOV_AFRI, BOV+AFRI, …)  
- **Profils de drogue (`drug_profile`)** :
  - Sensitive, Mono, MDR, Pre‑XDR, Other, Other*

---

## 🏗️ Structure du projet

```text
tuberculosis_prediction/
├── data/
│   ├── raw/
│   │   └── Afro_TB/
│   │       ├── 0-StartHERE_Afro-TB.xlsx      # Dataset principal
│   │       └── AFRO_TB_ANNOTATION_VCF/       # VCF annotés (mutations détaillées)
│   ├── processed/
│   │   ├── cleaned_dataset.csv               # Données nettoyées
│   │   ├── ml_ready_dataset.csv              # Dataset final pour ML
│   │   ├── mutation_features.csv             # (si mutations extraites)
│   │   ├── gene_features.csv                 # (optionnel)
│   │   └── drug_features.csv                 # (optionnel)
│   └── results/
│       ├── model_results.csv
│       ├── FINAL_REPORT.txt
│       ├── data_distribution.png
│       ├── feature_importance.png
│       ├── model_comparison.png
│       └── roc_curves.png
├── src/
│   ├── data_loader.py
│   ├── data_cleaner.py
│   ├── feature_extractor.py
│   ├── feature_selection.py
│   ├── models.py
│   ├── target_creator.py
│   ├── visualization.py
│   └── fitness_analysis.py
├── main.py
├── requirements.txt
└── notebooks/            # Pour analyses complémentaires
```

---

## ⚙️ Installation

```bash
cd tuberculosis_prediction

# Créer un environnement virtuel
python -m venv tb_env

# Activer l'environnement
# Windows
tb_env\Scripts\activate
# Linux/Mac
source tb_env/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

---

## 🚀 Exécution du pipeline

Assurez‑vous que le fichier Excel Afro‑TB est bien présent dans `data/raw/Afro_TB/0-StartHERE_Afro-TB.xlsx`.

```bash
cd tuberculosis_prediction
python main.py
```

Les résultats seront générés dans `data/results/` et un résumé dans `data/results/FINAL_REPORT.txt`.

---

## 🔄 Pipeline détaillé (`main.py`)

### 1. Chargement des données (`src/data_loader.py`)

- Lit `0-StartHERE_Afro-TB.xlsx` avec :
  - `sheet_name="AfroTB"`, `header=4`
- Renomme les colonnes :
  - `Name → sample_id`
  - `Country → country`
  - `Lineage → lineage`
  - `Drug → drug_profile`

### 2. Nettoyage (`src/data_cleaner.py`)

- Filtre uniquement les pays africains (26 pays).
- Supprime les doublons (sur `sample_id`).
- Normalise les noms de pays.
- Prépare des colonnes numériques potentielles pour les mutations (pour usage futur).
- Sauvegarde dans : `data/processed/cleaned_dataset.csv`.

### 3. Extraction de features (`src/feature_extractor.py`)

- Cherche des colonnes de mutations par gène (`rpoB`, `katG`, `inhA`, etc.).
- Si aucune colonne explicite n’est trouvée (cas actuel avec l’Excel Afro‑TB), tente une extraction « alternative » à partir des colonnes binaires/numériques.
- Produit éventuellement :
  - `mutation_features.csv`
  - `gene_features.csv` (compte de mutations par gène)
  - `drug_features.csv` (résistance par médicament)

> Remarque : les **vraies 157 mutations** sont dans les VCF annotés. Le pipeline actuel se base surtout sur les **profils de drogue**, pays et lignées.

### 4. Création des cibles (`src/target_creator.py`)

- À partir des (éventuelles) mutations, crée :
  - `resistant_rifampicin`, `resistant_isoniazid`, `resistant_ethambutol`,
    `resistant_pyrazinamide`, `resistant_fluoroquinolones`.
- Construit une colonne synthétique `resistance_profile` :
  - Sensitive, Mono‑resistant_RIF, MDR, XDR, Poly‑resistant_k, etc.

Dans ce projet, pour la tâche de ML, on crée aussi :

- `is_mdr`  (profil MDR)
- `is_xdr`  (profil XDR)
- `is_resistant` (MDR/Pre‑XDR/Other vs Sensitive/Mono)
- `is_sensitive`

### 5. Fusion des données

- Regroupe :
  - Métadonnées : `sample_id`, `country`, `lineage`
  - `mutation_features`, `gene_features`, `drug_features` (si présents)
  - Cibles : `resistance_profile`, `is_resistant`, etc.
- Sauvegarde : `data/processed/ml_ready_dataset.csv`.

### 6. Sélection de features (`src/feature_selection.py`)

- `remove_synonymous_features` :
  - Supprime les colonnes presque identiques (corrélation > 0.95).
- `select_highly_correlated_features` :
  - Garde les features les plus corrélées à `is_resistant`.
- Optionnel : `select_features_with_chi2`, `select_features_with_mutual_info`.

Au final, les features les plus importantes pour la prédiction sont :

- `is_mdr`
- `is_sensitive`
- `is_xdr` (peu informatif ici, car presque pas de XDR)

### 7. Entraînement des modèles (`src/models.py`)

- `prepare_ml_data` :
  - Split train/test (80 % / 20 %, stratifié).
  - Standardisation (`StandardScaler`).
- Modèles :
  - **Logistic Regression** (GridSearch sur C, penalty).
  - **KNN** (GridSearch sur k, metric, weights).
  - **Random Forest** (GridSearch sur n_estimators, profondeur, etc.).

### 8. Évaluation

- Validation croisée 5‑fold (`StratifiedKFold`).
- Métriques :
  - **Accuracy**
  - **F1‑Score (weighted)**
  - **AUC‑ROC**
- Résumé dans : `data/results/model_results.csv` et `data/results/FINAL_REPORT.txt`.

### 9. Visualisation (`src/visualization.py`)

- `data_distribution.png` :
  - Top 10 pays
  - Distribution des lignées
- `feature_importance.png` :
  - Importance des features (Random Forest)
- `model_comparison.png` :
  - Comparaison des modèles (CV vs test)
- `roc_curves.png` :
  - Courbes ROC pour chaque modèle

---

## 📊 Résultats principaux

Sur la tâche binaire **`is_resistant`** (résistant vs non‑résistant) :

- **Isolats analysés** : 13 691

**Performances sur le set de test :**

| Modèle              | Accuracy | F1‑Score | AUC‑ROC |
|---------------------|----------|---------|--------|
| Logistic Regression | 0.9127   | 0.9076  | 0.9715 |
| KNN                 | 0.9127   | 0.9076  | 0.9715 |
| Random Forest       | 0.9127   | 0.9076  | 0.9715 |

Features les plus importantes (Random Forest) :

- `is_mdr` (~0.58)
- `is_sensitive` (~0.42)
- `is_xdr` (~0.00 dans ce dataset)

---

## 🔬 Vers l’analyse des 157 mutations

- Les **157 mutations dans 12 gènes** sont disponibles dans les fichiers VCF annotés :
  - `data/raw/Afro_TB/AFRO_TB_ANNOTATION_VCF/`
- Le module `src/fitness_analysis.py` est prêt pour :
  - étudier les co‑occurrences de mutations,
  - trouver des mutations enrichies dans les MDR/XDR,
  - calculer un score de “fitness”.

Prochaine étape possible : parser les VCF pour ajouter ces mutations au `cleaned_dataset.csv` et relancer le pipeline.

---

## 📚 Références

- Laamarti M. et al. (2023), *Scientific Data* – « Afro‑TB dataset as a large scale genomic data of *Mycobacterium tuberculosis* in Africa ».  
- Hassan Oubrahim (2024), PFE : « Unraveling the Fitness Mechanism Of Mycobacterium Tuberculosis Based On An African Genomic Dataset ».


