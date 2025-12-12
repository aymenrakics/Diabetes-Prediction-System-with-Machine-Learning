import pandas as pd

class DataLoader:
    def __init__(self, chemin = "./data/diabetes_binary_health_indicators_BRFSS2015.csv"):
        self.data = pd.read_csv(chemin)
    
    def afficher_nb_lignes_colonnes(self):        
        print(f"Nombre des lignes = {self.data.shape[0]}")
        print(f"Nombre des colonnes = {self.data.shape[1]}")
    
    def extrait_variable_explicatif(self, variable_explicatif = "Diabetes_Status"):        
        return self.data[variable_explicatif]
    
    def extrait_variables_predicteurs(self):        
        df_prime = self.data.drop(columns = "Diabetes_Status", axis = 1)
        return df_prime