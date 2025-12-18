"""
Classe pour nettoyer et valider les données Afro-TB
"""
import pandas as pd
import numpy as np

class DataCleaner:
    """
    Classe pour nettoyer et valider les données Afro-TB
    """
    
    # Pays africains selon l'article
    AFRICAN_COUNTRIES = [
        'Algeria', 'Botswana', 'Cameroon', 'Ivory Coast', 'Congo',
        'Djibouti', 'Eswatini', 'Ethiopia', 'Gambia', 'Ghana',
        'Guinea', 'Kenya', 'Liberia', 'Madagascar', 'Malawi',
        'Mali', 'Mozambique', 'Nigeria', 'Rwanda', 'Sierra Leone',
        'South Africa', 'Sudan', 'Tanzania', 'Tunisia', 'Uganda',
        'Zimbabwe'
    ]
    
    # Gènes de résistance selon le PFE
    RESISTANCE_GENES = [
        'rpoB', 'katG', 'inhA', 'embB', 'pncA', 
        'gyrA', 'gyrB', 'rrs', 'tlyA', 'eis', 
        'thyA', 'folC', 'ddn'
    ]
    
    def __init__(self, df):
        self.df = df.copy()
        self.cleaning_log = []
    
    def clean_dataset(self):
        """
        Pipeline de nettoyage complet
        """
        print("Starting data cleaning...")
        initial_count = len(self.df)
        
        # 1. Filtrer par contamination
        self._filter_contamination()
        
        # 2. Filtrer par pays africains
        self._filter_african_countries()
        
        # 3. Supprimer les doublons
        self._remove_duplicates()
        
        # 4. Nettoyer les mutations
        self._clean_mutations()
        
        # 5. Valider les données
        self._validate_data()
        
        final_count = len(self.df)
        removed = initial_count - final_count
        
        print(f"\nCleaning completed:")
        print(f"  Initial: {initial_count} isolates")
        print(f"  Final: {final_count} isolates")
        print(f"  Removed: {removed} isolates ({removed/initial_count*100:.2f}%)")
        
        return self.df
    
    def _filter_contamination(self):
        """Filtrer les souches avec contamination < 10%"""
        if 'contamination' in self.df.columns:
            before = len(self.df)
            self.df = self.df[self.df['contamination'] < 0.10]
            after = len(self.df)
            self.cleaning_log.append(f"Contamination filter: {before} -> {after}")
    
    def _filter_african_countries(self):
        """Filtrer uniquement les pays africains"""
        if 'country' in self.df.columns:
            before = len(self.df)
            # Normaliser les noms de pays
            self.df['country'] = self.df['country'].astype(str).str.strip()
            # Essayer différentes variantes de noms
            country_mapping = {
                'Côte d\'Ivoire': 'Ivory Coast',
                'Ivory Coast': 'Ivory Coast',
                'Cote d\'Ivoire': 'Ivory Coast',
            }
            self.df['country'] = self.df['country'].replace(country_mapping)
            self.df = self.df[self.df['country'].isin(self.AFRICAN_COUNTRIES)]
            after = len(self.df)
            self.cleaning_log.append(f"African countries filter: {before} -> {after}")
    
    def _remove_duplicates(self):
        """Supprimer les doublons"""
        before = len(self.df)
        if 'sample_id' in self.df.columns:
            self.df = self.df.drop_duplicates(subset=['sample_id'])
        else:
            # Essayer d'autres colonnes d'ID
            id_cols = [col for col in self.df.columns if 'id' in col.lower() or 'accession' in col.lower()]
            if id_cols:
                self.df = self.df.drop_duplicates(subset=id_cols[0])
            else:
                self.df = self.df.drop_duplicates()
        after = len(self.df)
        self.cleaning_log.append(f"Duplicates removed: {before} -> {after}")
    
    def _clean_mutations(self):
        """Nettoyer les colonnes de mutations"""
        # Identifier les colonnes de mutations
        mutation_cols = [col for col in self.df.columns 
                        if any(gene.lower() in col.lower() for gene in self.RESISTANCE_GENES)]
        
        # Si pas de colonnes trouvées avec les noms de gènes, chercher autrement
        if len(mutation_cols) == 0:
            # Chercher toutes les colonnes numériques qui pourraient être des mutations
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclure les colonnes de métadonnées
            exclude_cols = ['contamination', 'sample_id', 'id']
            mutation_cols = [col for col in numeric_cols 
                           if not any(exc in col.lower() for exc in exclude_cols)]
        
        print(f"\nFound {len(mutation_cols)} mutation columns")
        
        # Remplacer les valeurs manquantes par 0 (pas de mutation)
        for col in mutation_cols:
            # Si la colonne contient des valeurs non-numériques, convertir
            if self.df[col].dtype == 'object':
                # Remplacer 'WT', 'wild-type', etc. par 0
                self.df[col] = self.df[col].replace(['WT', 'wild-type', 'Wild-type', '', 'nan', 'NaN'], 0)
                # Convertir en numérique
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Remplacer NaN par 0
            self.df[col] = self.df[col].fillna(0)
            
            # Convertir en binaire: 0 = pas de mutation, 1 = mutation présente
            # Garder les valeurs numériques si elles existent déjà
            if self.df[col].dtype in ['int64', 'float64']:
                self.df[col] = (self.df[col] != 0).astype(int)
    
    def _validate_data(self):
        """Valider l'intégrité des données"""
        errors = []
        
        # Vérifier qu'on a au moins quelques milliers de souches
        if len(self.df) < 1000:
            errors.append(f"Warning: Only {len(self.df)} isolates, expected ~13,753")
        
        # Vérifier la présence des gènes de résistance
        mutation_cols = [col for col in self.df.columns 
                        if any(gene.lower() in col.lower() for gene in self.RESISTANCE_GENES)]
        
        if len(mutation_cols) < 10:  # Au moins 10 mutations attendues
            errors.append(f"Warning: Only {len(mutation_cols)} mutation columns found")
        
        if errors:
            print("\nValidation Warnings:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("\n✓ Data validation passed!")

