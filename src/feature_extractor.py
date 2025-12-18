"""
Extraire les features de mutations pour l'apprentissage machine
"""
import pandas as pd
import numpy as np

class MutationFeatureExtractor:
    """
    Extraire les features de mutations pour l'apprentissage machine
    """
    
    # Mapping gènes -> médicaments selon le PFE
    GENE_DRUG_MAPPING = {
        'rpoB': 'Rifampicin',
        'katG': 'Isoniazid',
        'inhA': 'Isoniazid',
        'embB': 'Ethambutol',
        'pncA': 'Pyrazinamide',
        'gyrA': 'Fluoroquinolones',
        'gyrB': 'Fluoroquinolones',
        'rrs': 'Aminoglycosides',
        'tlyA': 'Capreomycin',
        'eis': 'Kanamycin',
        'thyA': 'PAS',
        'folC': 'PAS',
        'ddn': 'Pretomanid'
    }
    
    # Mutations connues importantes (exemples)
    KNOWN_MUTATIONS = {
        'rpoB': ['S450L', 'D435V', 'H445D', 'L430P'],
        'katG': ['S315T', 'S315N', 'S315I'],
        'inhA': ['S94A', 'I21V', 'I16T'],
        'embB': ['M306V', 'M306I', 'M306L'],
        'pncA': ['H51D', 'D63G', 'V139A'],
        'gyrA': ['D94G', 'A90V', 'S91P'],
        'gyrB': ['E540D', 'D500H', 'N538D'],
        'rrs': ['A1401G', 'C1402T', 'G1484T'],
    }
    
    def __init__(self, df):
        self.df = df.copy()
        self.mutation_features = None
    
    def extract_mutation_features(self):
        """
        Extraire toutes les features de mutations
        """
        mutation_cols = []
        
        # Identifier toutes les colonnes de mutations
        for gene in self.GENE_DRUG_MAPPING.keys():
            gene_cols = [col for col in self.df.columns 
                        if gene.lower() in col.lower() and 
                        ('mutation' in col.lower() or 'mut' in col.lower() or 
                         any(mut in col for mut in self.KNOWN_MUTATIONS.get(gene, [])))]
            mutation_cols.extend(gene_cols)
        
        # Si pas de colonnes trouvées avec les noms de gènes, utiliser toutes les colonnes numériques
        if len(mutation_cols) == 0:
            print("Warning: Mutation columns not found with standard naming")
            print("Attempting alternative extraction...")
            mutation_cols = self._extract_alternative()
        
        # Créer un DataFrame avec uniquement les mutations
        if mutation_cols:
            self.mutation_features = self.df[mutation_cols].copy()
            print(f"Extracted {len(mutation_cols)} mutation features")
        else:
            print("Warning: No mutation features found")
            self.mutation_features = pd.DataFrame()
        
        return self.mutation_features
    
    def _extract_alternative(self):
        """
        Méthode alternative pour extraire les mutations
        """
        # Chercher toutes les colonnes qui pourraient être des mutations
        potential_cols = []
        
        for col in self.df.columns:
            # Ignorer les colonnes de métadonnées
            if col.lower() in ['sample_id', 'country', 'lineage', 'sra_accession', 
                              'contamination', 'resistance_profile', 'id']:
                continue
            
            # Si la colonne est binaire ou numérique, considérer comme mutation potentielle
            if self.df[col].dtype in ['int64', 'float64', 'bool']:
                unique_vals = self.df[col].unique()
                # Si seulement 0 et 1 (ou similaire), c'est probablement une mutation
                if len(unique_vals) <= 3 and (0 in unique_vals or np.nan in unique_vals):
                    potential_cols.append(col)
        
        print(f"Found {len(potential_cols)} potential mutation columns")
        return potential_cols
    
    def create_gene_level_features(self):
        """
        Créer des features au niveau des gènes (nombre de mutations par gène)
        """
        gene_features = {}
        
        for gene in self.GENE_DRUG_MAPPING.keys():
            gene_cols = [col for col in self.mutation_features.columns 
                        if gene.lower() in col.lower()]
            
            if gene_cols:
                # Nombre total de mutations dans ce gène
                gene_features[f'{gene}_mutation_count'] = self.mutation_features[gene_cols].sum(axis=1)
                # Présence d'au moins une mutation
                gene_features[f'{gene}_has_mutation'] = (self.mutation_features[gene_cols].sum(axis=1) > 0).astype(int)
        
        if gene_features:
            gene_df = pd.DataFrame(gene_features)
            return gene_df
        else:
            return pd.DataFrame()
    
    def create_drug_resistance_features(self):
        """
        Créer des features au niveau des médicaments
        """
        drug_features = {}
        
        for gene, drug in self.GENE_DRUG_MAPPING.items():
            gene_cols = [col for col in self.mutation_features.columns 
                        if gene.lower() in col.lower()]
            
            if gene_cols:
                drug_key = drug.replace(' ', '_').lower()
                if drug_key not in drug_features:
                    drug_features[drug_key] = (self.mutation_features[gene_cols].sum(axis=1) > 0).astype(int)
                else:
                    # Si plusieurs gènes pour le même médicament (OR logique)
                    drug_features[drug_key] = (
                        drug_features[drug_key] | 
                        (self.mutation_features[gene_cols].sum(axis=1) > 0)
                    ).astype(int)
        
        if drug_features:
            drug_df = pd.DataFrame(drug_features)
            return drug_df
        else:
            return pd.DataFrame()

