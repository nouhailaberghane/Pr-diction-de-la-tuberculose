"""
Analyse des mutations liées à la "fitness" - nouvelles mutations potentielles
"""
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

class FitnessMutationAnalyzer:
    """
    Identifier de nouvelles mutations potentielles liées à la fitness
    en analysant les associations avec les profils de résistance
    """
    
    def __init__(self, df, resistance_genes):
        self.df = df.copy()
        self.resistance_genes = resistance_genes
        self.fitness_mutations = []
        
    def identify_fitness_mutations(self, target_col='resistance_profile', 
                                   min_support=0.01, min_confidence=0.3):
        """
        Identifier les mutations potentiellement liées à la fitness
        
        Critères:
        1. Mutations fréquentes dans les souches résistantes
        2. Mutations co-occurrentes avec des mutations de résistance connues
        3. Mutations associées statistiquement aux profils de résistance
        """
        print("="*60)
        print("FITNESS MUTATION ANALYSIS")
        print("="*60)
        
        # 1. Identifier toutes les colonnes de mutations
        mutation_cols = [col for col in self.df.columns 
                        if any(gene.lower() in col.lower() for gene in self.resistance_genes)]
        
        # Ajouter toutes les autres colonnes qui pourraient être des mutations
        all_mutation_cols = []
        for col in self.df.columns:
            if col in ['sample_id', 'country', 'lineage', 'resistance_profile', 
                      'resistant_rifampicin', 'resistant_isoniazid', 'resistant_ethambutol',
                      'resistant_pyrazinamide', 'resistant_fluoroquinolones']:
                continue
            if self.df[col].dtype in ['int64', 'float64']:
                unique_vals = self.df[col].unique()
                if len(unique_vals) <= 3 and (0 in unique_vals or 1 in unique_vals):
                    all_mutation_cols.append(col)
        
        print(f"\nTotal mutation columns found: {len(all_mutation_cols)}")
        
        # 2. Filtrer les mutations connues (déjà dans les 157 mutations)
        known_mutations = set(mutation_cols)
        potential_fitness_mutations = [col for col in all_mutation_cols 
                                      if col not in known_mutations]
        
        print(f"Known resistance mutations: {len(known_mutations)}")
        print(f"Potential fitness mutations to analyze: {len(potential_fitness_mutations)}")
        
        # 3. Analyser chaque mutation potentielle
        fitness_results = []
        
        if target_col not in self.df.columns:
            print(f"Warning: Target column '{target_col}' not found. Using resistance profiles.")
            # Créer une variable binaire pour MDR/XDR
            self.df['is_mdr_xdr'] = self.df['resistance_profile'].isin(['MDR', 'XDR']).astype(int)
            target_col = 'is_mdr_xdr'
        
        for mutation in potential_fitness_mutations[:200]:  # Limiter pour performance
            try:
                result = self._analyze_mutation_fitness(mutation, target_col, 
                                                       min_support, min_confidence)
                if result:
                    fitness_results.append(result)
            except Exception as e:
                continue
        
        # 4. Trier par score de fitness
        if fitness_results:
            fitness_df = pd.DataFrame(fitness_results)
            fitness_df = fitness_df.sort_values('fitness_score', ascending=False)
            
            print(f"\nFound {len(fitness_df)} potential fitness mutations")
            print("\nTop 20 fitness mutations:")
            print(fitness_df.head(20)[['mutation', 'frequency', 'mdr_xdr_association', 
                                      'fitness_score']].to_string())
            
            return fitness_df
        else:
            print("No fitness mutations identified")
            return pd.DataFrame()
    
    def _analyze_mutation_fitness(self, mutation, target_col, min_support, min_confidence):
        """
        Analyser une mutation individuelle pour son association avec la fitness
        """
        # Fréquence de la mutation
        mutation_freq = self.df[mutation].mean()
        
        if mutation_freq < min_support:
            return None
        
        # Association avec MDR/XDR ou profils de résistance
        if target_col in ['resistance_profile']:
            # Créer une variable binaire MDR/XDR
            is_resistant = self.df['resistance_profile'].isin(['MDR', 'XDR', 'Poly-resistant_2', 
                                                               'Poly-resistant_3', 'Poly-resistant_4']).astype(int)
        else:
            is_resistant = self.df[target_col]
        
        # Table de contingence
        contingency = pd.crosstab(self.df[mutation], is_resistant)
        
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return None
        
        # Test statistique (Chi-square ou Fisher)
        try:
            if contingency.min().min() < 5:
                # Utiliser Fisher exact test pour petits échantillons
                oddsratio, p_value = fisher_exact(contingency)
            else:
                # Utiliser Chi-square
                chi2, p_value, dof, expected = chi2_contingency(contingency)
        except:
            return None
        
        # Calculer l'association
        mutation_in_resistant = self.df[self.df[mutation] == 1][target_col].value_counts()
        mutation_in_sensitive = self.df[self.df[mutation] == 0][target_col].value_counts()
        
        # Score de fitness basé sur:
        # 1. Fréquence de la mutation
        # 2. Association avec résistance (p-value)
        # 3. Co-occurrence avec mutations connues
        
        # Calculer la co-occurrence avec mutations de résistance connues
        known_mut_cols = [col for col in self.df.columns 
                         if any(gene in col.lower() for gene in ['rpoB', 'katG', 'inhA', 'embB'])]
        cooccurrence_score = 0
        if known_mut_cols:
            cooccurrences = []
            for known_mut in known_mut_cols[:10]:  # Limiter pour performance
                if known_mut != mutation:
                    cooc = ((self.df[mutation] == 1) & (self.df[known_mut] == 1)).sum()
                    cooccurrences.append(cooc / max(self.df[mutation].sum(), 1))
            cooccurrence_score = np.mean(cooccurrences) if cooccurrences else 0
        
        # Score de fitness combiné
        fitness_score = (mutation_freq * 0.3 + 
                         (1 - p_value) * 0.4 + 
                         cooccurrence_score * 0.3)
        
        # Association MDR/XDR
        if target_col == 'resistance_profile':
            mdr_xdr_rate_with = self.df[(self.df[mutation] == 1) & 
                                       (self.df['resistance_profile'].isin(['MDR', 'XDR']))].shape[0] / max(self.df[mutation].sum(), 1)
            mdr_xdr_rate_without = self.df[(self.df[mutation] == 0) & 
                                          (self.df['resistance_profile'].isin(['MDR', 'XDR']))].shape[0] / max((self.df[mutation] == 0).sum(), 1)
            mdr_xdr_association = mdr_xdr_rate_with - mdr_xdr_rate_without
        else:
            mdr_xdr_association = 0
        
        if fitness_score > 0.1:  # Seuil minimum
            return {
                'mutation': mutation,
                'frequency': mutation_freq,
                'p_value': p_value,
                'cooccurrence_score': cooccurrence_score,
                'mdr_xdr_association': mdr_xdr_association,
                'fitness_score': fitness_score
            }
        
        return None
    
    def analyze_mutation_patterns(self, top_n=50):
        """
        Analyser les patterns de mutations pour identifier des combinaisons liées à la fitness
        """
        print("\n" + "="*60)
        print("MUTATION PATTERN ANALYSIS")
        print("="*60)
        
        # Identifier les mutations les plus fréquentes dans les souches MDR/XDR
        mdr_xdr_samples = self.df[self.df['resistance_profile'].isin(['MDR', 'XDR'])]
        
        if len(mdr_xdr_samples) == 0:
            print("No MDR/XDR samples found")
            return pd.DataFrame()
        
        # Calculer la fréquence des mutations dans MDR/XDR vs Sensitive
        mutation_cols = [col for col in self.df.columns 
                        if col not in ['sample_id', 'country', 'lineage', 'resistance_profile',
                                      'resistant_rifampicin', 'resistant_isoniazid', 
                                      'resistant_ethambutol', 'resistant_pyrazinamide',
                                      'resistant_fluoroquinolones']]
        
        mutation_freq_mdr = {}
        mutation_freq_sensitive = {}
        
        sensitive_samples = self.df[self.df['resistance_profile'] == 'Sensitive']
        
        for col in mutation_cols[:200]:  # Limiter pour performance
            if self.df[col].dtype in ['int64', 'float64']:
                try:
                    freq_mdr = mdr_xdr_samples[col].mean()
                    freq_sens = sensitive_samples[col].mean() if len(sensitive_samples) > 0 else 0
                    
                    if freq_mdr > 0.05:  # Au moins 5% dans MDR/XDR
                        mutation_freq_mdr[col] = freq_mdr
                        mutation_freq_sensitive[col] = freq_sens
                except:
                    continue
        
        # Créer un DataFrame avec les résultats
        pattern_df = pd.DataFrame({
            'mutation': list(mutation_freq_mdr.keys()),
            'frequency_mdr_xdr': list(mutation_freq_mdr.values()),
            'frequency_sensitive': [mutation_freq_sensitive.get(m, 0) for m in mutation_freq_mdr.keys()],
        })
        
        pattern_df['enrichment_ratio'] = (pattern_df['frequency_mdr_xdr'] / 
                                         (pattern_df['frequency_sensitive'] + 0.001))
        pattern_df = pattern_df.sort_values('enrichment_ratio', ascending=False)
        
        print(f"\nTop {top_n} mutations enriched in MDR/XDR:")
        print(pattern_df.head(top_n)[['mutation', 'frequency_mdr_xdr', 
                                    'frequency_sensitive', 'enrichment_ratio']].to_string())
        
        return pattern_df

