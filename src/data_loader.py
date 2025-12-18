"""
Classe pour charger et préparer le dataset Afro-TB
"""
import pandas as pd
import numpy as np
from pathlib import Path

class AfroTBDataLoader:
    """
    Classe pour charger et préparer le dataset Afro-TB
    """
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = None
        self.metadata = None
        
    def load_dataset(self):
        """
        Charger le dataset (supporte plusieurs formats)
        """
        # Essayer différents formats (inclure le fichier réel Afro-TB)
        possible_files = [
            'Afro_TB/0-StartHERE_Afro-TB.xlsx',  # Fichier réel Afro-TB
            'StartHERE_AFRO_TB.xlsx',  # Fichier de démo
            'afro_tb_dataset.csv',
            'afro_tb_data.csv',
            'metadata.csv',
            'mutations.csv',
            'resistance_profiles.csv'
        ]
        
        for filename in possible_files:
            filepath = self.data_path / filename
            if filepath.exists():
                print(f"Loading {filename}...")
                if filename.endswith('.xlsx'):
                    # Cas spécifique : vrai fichier Afro-TB officiel
                    if "Afro_TB/0-StartHERE_Afro-TB.xlsx" in filename:
                        # D'après l'exploration, la feuille s'appelle 'AfroTB'
                        # et la ligne d'en-tête utile est à l'index 4 (row 5 Excel)
                        print("Detected official Afro-TB Excel structure, using sheet='AfroTB', header=4")
                        self.df = pd.read_excel(filepath, sheet_name="AfroTB", header=4)

                        # Renommer les colonnes clés pour correspondre au reste du pipeline
                        rename_map = {}
                        for col in self.df.columns:
                            if str(col).strip().lower() == "name":
                                rename_map[col] = "sample_id"
                            elif str(col).strip().lower() == "country":
                                rename_map[col] = "country"
                            elif str(col).strip().lower() == "lineage":
                                rename_map[col] = "lineage"
                            elif str(col).strip().lower() == "drug":
                                rename_map[col] = "drug_profile"
                        if rename_map:
                            self.df = self.df.rename(columns=rename_map)

                        # Nettoyer les lignes entièrement vides
                        self.df = self.df.dropna(axis=0, how="all")
                        # Nettoyer les colonnes entièrement vides
                        self.df = self.df.dropna(axis=1, how="all")
                    else:
                        # Fichiers Excel génériques (ex : dataset de démo)
                        try:
                            # D'abord, vérifier les feuilles
                            xl_file = pd.ExcelFile(filepath)
                            print(f"Available sheets: {xl_file.sheet_names}")

                            # Charger la première feuille avec header=0
                            self.df = pd.read_excel(filepath, sheet_name=0, header=0)

                            # Si la première colonne ressemble à une ligne de méta-données
                            if self.df.columns[0] == 'AFRO-TB dataset' or 'Unnamed' in str(self.df.columns[0]):
                                print("Detected meta row, trying with header=1...")
                                self.df = pd.read_excel(filepath, sheet_name=0, header=1)

                            # Nettoyer les colonnes/lignes vides
                            self.df = self.df.dropna(axis=1, how='all')
                            self.df = self.df.dropna(axis=0, how='all')

                        except Exception as e:
                            print(f"Error loading Excel: {e}")
                            # Fallback : méthode simple
                            self.df = pd.read_excel(filepath)
                else:
                    self.df = pd.read_csv(filepath)
                print(f"Loaded {len(self.df)} rows, {len(self.df.columns)} columns")
                return self.df
        
        # Si plusieurs fichiers séparés
        if (self.data_path / 'metadata.csv').exists():
            self._load_separate_files()
            return self.df
        
        raise FileNotFoundError(f"No data files found in {self.data_path}")
    
    def _load_separate_files(self):
        """
        Charger si les données sont dans plusieurs fichiers
        """
        metadata = pd.read_csv(self.data_path / 'metadata.csv')
        mutations = pd.read_csv(self.data_path / 'mutations.csv')
        resistance = pd.read_csv(self.data_path / 'resistance_profiles.csv')
        
        # Fusionner les fichiers
        self.df = metadata.merge(mutations, on='sample_id', how='inner')
        self.df = self.df.merge(resistance, on='sample_id', how='inner')
        
        print(f"Merged dataset: {len(self.df)} rows")
    
    def explore_dataset(self):
        """
        Exploration détaillée du dataset
        """
        if self.df is None:
            self.load_dataset()
        
        print("="*60)
        print("DATASET EXPLORATION")
        print("="*60)
        
        # Informations de base
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Colonnes
        print(f"\nColumns ({len(self.df.columns)}):")
        for i, col in enumerate(self.df.columns[:20], 1):  # Afficher les 20 premières
            print(f"  {i}. {col}")
        if len(self.df.columns) > 20:
            print(f"  ... and {len(self.df.columns) - 20} more columns")
        
        # Types de données
        print("\nData Types:")
        print(self.df.dtypes.value_counts())
        
        # Valeurs manquantes
        print("\nMissing Values:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_pct
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        if len(missing_df) > 0:
            print(missing_df.head(10))
        else:
            print("  No missing values found")
        
        # Statistiques descriptives
        print("\nDescriptive Statistics (numeric columns):")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(self.df[numeric_cols].describe())
        
        # Distribution des pays
        if 'country' in self.df.columns:
            print("\nCountry Distribution:")
            print(self.df['country'].value_counts().head(10))
        
        # Distribution des lignées
        if 'lineage' in self.df.columns:
            print("\nLineage Distribution:")
            print(self.df['lineage'].value_counts())
        
        # Distribution des profils de résistance
        if 'resistance_profile' in self.df.columns:
            print("\nResistance Profile Distribution:")
            print(self.df['resistance_profile'].value_counts())
        
        return self.df

