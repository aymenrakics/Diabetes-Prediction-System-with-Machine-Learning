from data_loader import DataLoader
from data_preprocessing import Preprocessing
from data_visualization import DataVisualizer
from model_tester import ModelTester
from file_manager import FileManager
from modele_evaluation import Evaluation

loader = DataLoader()
loader.afficher_nb_lignes_colonnes()
    
X = loader.extrait_variables_predicteurs()
y = loader.extrait_variable_explicatif()
            
prep = Preprocessing(X, y)
prep.traitement_des_valeurs_manquantes()
X_train, X_test, y_train, y_test = prep.separation_des_donnees()
            
viz = DataVisualizer(loader.data)
viz.distribution_age()
viz.distribution_imc()
viz.repartition_diabete()
viz.relation_age_imc()
            
tester = ModelTester(X_train, X_test, y_train, y_test)
modeles = tester.creer_et_entrainer_tous_modeles()
            
evaluateur = Evaluation(None, X_test)
resultats = evaluateur.evaluer_plusieurs_modeles(modeles, X_test, y_test)
classement = evaluateur.afficher_classement(resultats)
        
gestionnaire = FileManager('resultats.txt')
gestionnaire.sauvegarder_resultats(resultats, classement)    