from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

class ModelTester:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.modeles = {}  

    def creer_modele(self, modele="logistic_regression"):
        if modele == "logistic_regression":
            return LogisticRegression(max_iter=1000)
        elif modele == "decision_tree":
            return DecisionTreeClassifier()
        elif modele == "knn":
            return KNeighborsClassifier()
        elif modele == "svm":
            return svm.SVC()
        else:
            raise ValueError("Modèle inconnu !")

    def creer_et_entrainer_tous_modeles(self):
        modeles_liste = ["logistic_regression", "decision_tree", "knn", "svm"]
        noms = ["Régression Logistique", "Arbre de Décision", "KNN", "SVM"]

        for nom, modele_type in zip(noms, modeles_liste):
            modele = self.creer_modele(modele_type)
            modele.fit(self.X_train, self.y_train)
            self.modeles[nom] = modele

        return self.modeles
