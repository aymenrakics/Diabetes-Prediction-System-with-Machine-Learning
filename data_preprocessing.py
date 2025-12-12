import pandas as pd
from sklearn.model_selection import train_test_split

class Preprocessing:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def verifier_valeurs_manquantes(self):        
        if self.X.isnull().sum().sum() > 0:
            return True
        return False
    
    def traitement_des_valeurs_manquantes(self):        
        if self.verifier_valeurs_manquantes():
            self.X = self.X.fillna(self.X.mean())
    
    def separation_des_donnees(self, test_size = 0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size = test_size, random_state = 42
        )
        return X_train, X_test, y_train, y_test