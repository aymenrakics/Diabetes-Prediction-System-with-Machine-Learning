from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class Evaluation:
    def __init__(self, modele, X_test):        
        self.modele = modele
        self.X_test = X_test
        self.accuracy = None
        self.confusion_matrice = None
        self.classification_report_result = None
    
    def evaluer_modele(self, y_true):        
        predictions = self.modele.predict(self.X_test)
        self.accuracy = accuracy_score(y_true, predictions)
        self.confusion_matrice = confusion_matrix(y_true, predictions)
        self.classification_report_result = classification_report(y_true, predictions)
        
        return self.accuracy, self.confusion_matrice, self.classification_report_result
    
    def evaluer_plusieurs_modeles(self, modeles, X_test, y_test):
        resultats = {}
        
        for nom, modele in modeles.items():
            predictions = modele.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            cm = confusion_matrix(y_test, predictions)
            rapport = classification_report(y_test, predictions)
            
            resultats[nom] = {
                'accuracy': acc,
                'confusion_matrix': cm,
                'rapport': rapport
            }
            print(f"{nom} évalué - Accuracy: {acc:.4f}")
        
        return resultats
        
    def afficher_classement(self, resultats):                        
        classement = sorted(resultats.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for i, (nom, res) in enumerate(classement, 1):
            print(f"{i}. {nom}: {res['accuracy']:.4f}")
        
        return classement