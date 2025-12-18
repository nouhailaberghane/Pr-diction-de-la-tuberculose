"""
Créer les variables cibles (targets) pour la prédiction
"""
import pandas as pd
import numpy as np

class ResistanceProfileCreator:
    """
    Créer les variables cibles (targets) pour la prédiction
    """
    
    def __init__(self, df, mutation_features):
        self.df = df.copy()
        self.mutation_features = mutation_features.copy()
    
    def create_resistance_profiles(self):
        """
        Créer les profils de résistance selon les mutations
        """
        # Mapping gènes -> médicaments
        gene_drug_map = {
            'rpoB': 'rifampicin',
            'katG': 'isoniazid',
            'inhA': 'isoniazid',
            'embB': 'ethambutol',
            'pncA': 'pyrazinamide',
            'gyrA': 'fluoroquinolones',
            'gyrB': 'fluoroquinolones',
        }
        
        # Créer des colonnes binaires pour chaque médicament
        for drug in ['rifampicin', 'isoniazid', 'ethambutol', 'pyrazinamide', 'fluoroquinolones']:
            drug_cols = []
            for gene, gene_drug in gene_drug_map.items():
                if gene_drug == drug:
                    gene_cols = [col for col in self.mutation_features.columns 
                                if gene.lower() in col.lower()]
                    drug_cols.extend(gene_cols)
            
            if drug_cols:
                # Résistant si au moins une mutation présente
                self.df[f'resistant_{drug}'] = (
                    self.mutation_features[drug_cols].sum(axis=1) > 0
                ).astype(int)
            else:
                # Si pas de colonnes trouvées, créer une colonne vide
                self.df[f'resistant_{drug}'] = 0
        
        # Créer le profil de résistance global
        self.df['resistance_profile'] = self._classify_resistance_profile()
        
        return self.df
    
    def _classify_resistance_profile(self):
        """
        Classifier selon les définitions WHO:
        - Sensitive: pas de résistance
        - Mono-resistant: résistant à un seul médicament
        - MDR: résistant à Rifampicin + Isoniazid
        - XDR: MDR + résistant aux Fluoroquinolones + Aminoglycosides
        """
        profiles = []
        
        for idx, row in self.df.iterrows():
            resistant_drugs = []
            
            if row.get('resistant_rifampicin', 0) == 1:
                resistant_drugs.append('RIF')
            if row.get('resistant_isoniazid', 0) == 1:
                resistant_drugs.append('INH')
            if row.get('resistant_ethambutol', 0) == 1:
                resistant_drugs.append('EMB')
            if row.get('resistant_pyrazinamide', 0) == 1:
                resistant_drugs.append('PZA')
            if row.get('resistant_fluoroquinolones', 0) == 1:
                resistant_drugs.append('FQ')
            
            num_resistant = len(resistant_drugs)
            
            if num_resistant == 0:
                profiles.append('Sensitive')
            elif num_resistant == 1:
                profiles.append(f'Mono-resistant_{resistant_drugs[0]}')
            elif 'RIF' in resistant_drugs and 'INH' in resistant_drugs:
                # MDR ou XDR
                if 'FQ' in resistant_drugs:
                    profiles.append('XDR')
                else:
                    profiles.append('MDR')
            else:
                profiles.append(f'Poly-resistant_{num_resistant}')
        
        return profiles

